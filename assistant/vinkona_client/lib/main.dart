import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const VinkonaApp());
}

class VinkonaApp extends StatelessWidget {
  const VinkonaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vinkona',
      theme: ThemeData.dark(useMaterial3: true),
      home: const VoiceScreen(),
    );
  }
}

class VoiceScreen extends StatefulWidget {
  const VoiceScreen({super.key});

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen> {
  // ── Configuration ──────────────────────────────────────────────────────────
  // For WireGuard: use the WG interface IP (e.g. 10.0.0.1).
  // For LAN testing: use the host machine's LAN IP.
  static const String _serverIp = '192.168.1.50';
  static const int _serverPort = 8998;
  static const int _sampleRate = 24000;

  // ── State ──────────────────────────────────────────────────────────────────
  WebSocket? _ws;
  StreamSubscription? _wsSub;
  // Capture and playback share ONE flutter_sound session (full-duplex play+record),
  // which avoids the audio-session conflict that blocked the separate `record` plugin.
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _recorderOpen = false;
  StreamController<Uint8List>? _micController;
  StreamSubscription<Uint8List>? _recSub;
  final FlutterSoundPlayer _player = FlutterSoundPlayer();
  bool _playerOpen = false;

  // DEBUG: mic send telemetry
  int _micFrames = 0;
  int _micBytesSent = 0;

  // DEBUG: server-audio receive telemetry (verifies the server's pacing)
  int _rxAudioFrames = 0;
  int _rxAudioBytes = 0;
  DateTime? _rxStart;

  bool _isConnected = false;
  String _status = 'Ready';
  String _transcript = '';
  // Transcript is bounded + UI updates are throttled, so a flood of text tokens
  // can't blow up into O(n²) string churn (the GC-thrash culprit).
  static const int _maxTranscriptChars = 2000;
  bool _transcriptDirty = false;
  Timer? _transcriptTimer;

  // Whether Vinkona is exempt from battery optimization (reflected by the switch).
  bool _ignoreBatteryOpt = false;

  @override
  void initState() {
    super.initState();
    _initForegroundService();
    _refreshBatteryOptStatus();
  }

  // Reflect the OS-level battery-optimization exemption in the switch.
  Future<void> _refreshBatteryOptStatus() async {
    final ignoring = await FlutterForegroundTask.isIgnoringBatteryOptimizations;
    if (mounted) setState(() => _ignoreBatteryOpt = ignoring);
  }

  Future<void> _onBatteryOptToggle(bool value) async {
    if (value) {
      // Requests the exemption — a system dialog appears for the user to allow it.
      await FlutterForegroundTask.requestIgnoreBatteryOptimization();
    } else {
      // Android has no API to re-enable optimization; open Settings so the user
      // can turn it back on manually.
      await FlutterForegroundTask.openIgnoreBatteryOptimizationSettings();
    }
    // Re-read the real OS state (the user may have cancelled the dialog).
    await _refreshBatteryOptStatus();
  }

  @override
  void dispose() {
    _doDisconnect();
    super.dispose();
  }

  // ── Foreground service (hands-free background operation) ─────────────────────
  // Keeps the process, microphone, audio session, and WebSocket alive when the
  // screen locks. Without a foreground service, Android suspends all of these.
  void _initForegroundService() {
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'vinkona_live_call',
        channelName: 'Vinkona Live Call',
        channelDescription: 'Keeps the voice connection active while in the background.',
        channelImportance: NotificationChannelImportance.LOW,
        priority: NotificationPriority.LOW,
      ),
      iosNotificationOptions: const IOSNotificationOptions(),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.nothing(), // no periodic task; just keep-alive
        allowWakeLock: true, // keep CPU running with the screen off
        allowWifiLock: true, // keep Wi-Fi out of power-save so the socket doesn't stall
      ),
    );
  }

  Future<void> _startForegroundService() async {
    // POST_NOTIFICATIONS is a runtime permission on Android 13+.
    if (await FlutterForegroundTask.checkNotificationPermission() !=
        NotificationPermission.granted) {
      await FlutterForegroundTask.requestNotificationPermission();
    }
    if (await FlutterForegroundTask.isRunningService) return;
    await FlutterForegroundTask.startService(
      serviceId: 1001,
      notificationTitle: 'Vinkona is live',
      notificationText: 'Tap to return to the conversation.',
    );
  }

  Future<void> _stopForegroundService() async {
    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.stopService();
    }
  }

  // ── Audio conversion helpers ───────────────────────────────────────────────

  // PCM-16 little-endian bytes → float32 wire frame (kind 0x03 payload)
  // Reads via ByteData so unaligned buffer views (the recorder hands us views
  // at arbitrary offsets) and endianness are handled explicitly.
  Uint8List _pcm16ToF32Frame(Uint8List pcm16Bytes) {
    final src = ByteData.sublistView(pcm16Bytes);
    final n = pcm16Bytes.lengthInBytes ~/ 2;
    final out = Uint8List(1 + n * 4);
    out[0] = 0x03;
    final dst = ByteData.sublistView(out, 1);
    for (int i = 0; i < n; i++) {
      final s = src.getInt16(i * 2, Endian.little);
      dst.setFloat32(i * 4, s / 32768.0, Endian.little);
    }
    return out;
  }

  // Incoming float32 payload → PCM-16 bytes for flutter_sound
  Uint8List _f32PayloadToPcm16(Uint8List payload) {
    final src = ByteData.sublistView(payload);
    final n = payload.lengthInBytes ~/ 4;
    final out = Uint8List(n * 2);
    final dst = ByteData.sublistView(out);
    for (int i = 0; i < n; i++) {
      final f = src.getFloat32(i * 4, Endian.little).clamp(-1.0, 1.0);
      dst.setInt16(i * 2, (f * 32767.0).round(), Endian.little);
    }
    return out;
  }

  // ── Connection lifecycle ───────────────────────────────────────────────────

  Future<void> _connect() async {
    if (!await Permission.microphone.request().isGranted) {
      setState(() => _status = 'Microphone permission denied');
      return;
    }
    setState(() => _status = 'Connecting…');

    try {
      // Accept self-signed cert; replace with proper cert verification in production.
      final httpClient = HttpClient()
        ..badCertificateCallback = (cert, host, port) => true;

      final ws = await WebSocket.connect(
        'wss://$_serverIp:$_serverPort/api/chat',
        customClient: httpClient,
      );
      _ws = ws;

      // Wait for the server handshake byte (0x00 = ready)
      final ready = Completer<void>();
      bool handshakeDone = false;

      _wsSub = ws.listen(
        (dynamic data) {
          if (!handshakeDone) {
            if (data is List<int> && data.isNotEmpty && data[0] == 0x00) {
              handshakeDone = true;
              ready.complete();
            }
            return;
          }
          _onServerMessage(data);
        },
        onDone: () {
          if (!ready.isCompleted) ready.completeError('WebSocket closed');
          _doDisconnect();
        },
        onError: (e) {
          if (!ready.isCompleted) ready.completeError(e);
          _doDisconnect();
        },
      );

      await ready.future.timeout(const Duration(seconds: 10));

      // Open BOTH the player and recorder on the same session before starting
      // either stream, so the platform session comes up as play+record (duplex).
      await _player.openPlayer();
      _playerOpen = true;
      await _recorder.openRecorder();
      _recorderOpen = true;

      await _player.startPlayerFromStream(
        codec: Codec.pcm16,
        interleaved: true,
        numChannels: 1,
        sampleRate: _sampleRate,
        bufferSize: 8192,
      );

      // Prime the player with 200 ms of silence so the buffer is never empty
      // when the first real audio frame arrives — prevents underflow glitches
      // during the initial handshake and during quiet/listening periods.
      _player.uint8ListSink?.add(Uint8List(_sampleRate ~/ 5 * 2)); // 200ms × 2 bytes/sample

      setState(() {
        _isConnected = true;
        _status = 'Live';
        _transcript = '';
      });

      // Microphone → server streaming, via flutter_sound's PCM-16 stream sink.
      // NOTE: platform voice processing (AEC/NS) is OFF on purpose — on many
      // Android devices it forces the session to 16 kHz, which the server then
      // misreads as 24 kHz (speech sounds like noise). With it off we get a
      // clean 24 kHz capture; use HEADPHONES to avoid speaker→mic feedback,
      // exactly as the reference moshi client does.
      _micController = StreamController<Uint8List>();
      _recSub = _micController!.stream.listen(_onMicData);
      await _recorder.startRecorder(
        toStream: _micController!.sink,
        codec: Codec.pcm16,
        numChannels: 1,
        sampleRate: _sampleRate,
        enableVoiceProcessing: false,
        enableEchoCancellation: false,
        enableNoiseSuppression: false,
      );

      // Promote to a foreground service now that we're live and in the foreground,
      // so the call survives screen lock / backgrounding (hands-free operation).
      await _startForegroundService();
    } catch (e) {
      setState(() => _status = 'Error: $e');
      await _doDisconnect();
    }
  }

  void _onMicData(Uint8List pcm16Bytes) {
    if (_ws == null || pcm16Bytes.isEmpty) return;

    // DEBUG: measure what the mic is actually producing (peak amplitude 0..32767)
    final bd = ByteData.sublistView(pcm16Bytes);
    final n = pcm16Bytes.lengthInBytes ~/ 2;
    int peak = 0;
    for (int i = 0; i < n; i++) {
      final s = bd.getInt16(i * 2, Endian.little).abs();
      if (s > peak) peak = s;
    }

    final frame = _pcm16ToF32Frame(pcm16Bytes);
    _ws!.add(frame);

    _micFrames++;
    _micBytesSent += frame.length;
    if (_micFrames == 1 || _micFrames % 25 == 0) {
      debugPrint('🎙️ mic#$_micFrames in=${pcm16Bytes.lengthInBytes}B '
          'out=${frame.length}B peak=$peak totalSent=$_micBytesSent');
    }
  }

  void _onServerMessage(dynamic data) {
    if (data is! List<int> || data.isEmpty) return;
    // Avoid copying when the WebSocket already hands us a Uint8List.
    final msg = data is Uint8List ? data : Uint8List.fromList(data);
    final kind = msg[0];
    final payload = Uint8List.sublistView(msg, 1);

    if (kind == 0x03) {
      // Raw float32 PCM from server — convert to PCM16 and feed to player
      _rxStart ??= DateTime.now();
      _rxAudioFrames++;
      _rxAudioBytes += payload.length;
      if (_rxAudioFrames % 25 == 0) {
        final secs = DateTime.now().difference(_rxStart!).inMilliseconds / 1000.0;
        final kbps = secs > 0 ? _rxAudioBytes / 1024 / secs : 0;
        debugPrint('🔊 rx#$_rxAudioFrames payload=${payload.length}B '
            'avg=${kbps.toStringAsFixed(1)}KB/s');
      }
      final pcm16 = _f32PayloadToPcm16(payload);
      _player.uint8ListSink?.add(pcm16);
    } else if (kind == 0x02) {
      _appendTranscript(String.fromCharCodes(payload));
    }
  }

  // Bounded append + throttled repaint (≈10 Hz). Prevents unbounded string
  // growth and per-token setState rebuilds from thrashing the GC.
  void _appendTranscript(String text) {
    var t = _transcript + text;
    if (t.length > _maxTranscriptChars) {
      t = t.substring(t.length - _maxTranscriptChars);
    }
    _transcript = t;
    _transcriptDirty = true;
    _transcriptTimer ??=
        Timer.periodic(const Duration(milliseconds: 100), (_) {
      if (_transcriptDirty && mounted) {
        _transcriptDirty = false;
        setState(() {});
      }
    });
  }

  Future<void> _doDisconnect() async {
    await _stopForegroundService();

    _transcriptTimer?.cancel();
    _transcriptTimer = null;

    await _recSub?.cancel();
    _recSub = null;
    if (_recorderOpen) {
      if (_recorder.isRecording) await _recorder.stopRecorder();
      await _recorder.closeRecorder();
      _recorderOpen = false;
    }
    await _micController?.close();
    _micController = null;

    await _wsSub?.cancel();
    _wsSub = null;
    await _ws?.close();
    _ws = null;

    if (_playerOpen) {
      if (_player.isPlaying) {
        await _player.stopPlayer();
      }
      await _player.closePlayer();
      _playerOpen = false;
    }

    if (mounted) {
      setState(() {
        _isConnected = false;
        _status = 'Disconnected';
      });
    }
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Vinkona')),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Status indicator
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _isConnected ? Colors.greenAccent : Colors.grey,
                  ),
                ),
                const SizedBox(width: 8),
                Text(_status, style: const TextStyle(fontSize: 16)),
              ],
            ),
            const SizedBox(height: 24),

            // Live transcript
            if (_transcript.isNotEmpty)
              Container(
                width: double.infinity,
                constraints: const BoxConstraints(maxHeight: 200),
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white10,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: SingleChildScrollView(
                  child: Text(
                    _transcript,
                    style: const TextStyle(fontSize: 15),
                    textAlign: TextAlign.center,
                  ),
                ),
              )
            else
              const SizedBox(height: 60),

            const SizedBox(height: 48),

            // Connect / disconnect button
            SizedBox(
              width: 200,
              height: 56,
              child: ElevatedButton(
                onPressed: _isConnected ? _doDisconnect : _connect,
                style: ElevatedButton.styleFrom(
                  backgroundColor:
                      _isConnected ? Colors.redAccent : Colors.green,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(28),
                  ),
                ),
                child: Text(
                  _isConnected ? 'Disconnect' : 'Connect',
                  style: const TextStyle(fontSize: 18, color: Colors.white),
                ),
              ),
            ),

            const SizedBox(height: 32),

            // Battery-optimization exemption (helps the call survive screen-off
            // on aggressive devices). User-controlled; a system dialog may appear.
            SwitchListTile(
              value: _ignoreBatteryOpt,
              onChanged: _onBatteryOptToggle,
              contentPadding: EdgeInsets.zero,
              title: const Text('Keep alive in background'),
              subtitle: const Text(
                'Exempts Vinkona from battery optimization so hands-free calls '
                'survive screen lock. A system dialog may appear.',
                style: TextStyle(fontSize: 12),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
