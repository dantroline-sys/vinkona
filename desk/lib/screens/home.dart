/// Home: is she there, and what is she doing?
///
/// One connection card and one activity card, polled gently.  When the backend
/// is down this is the screen that says so in plain words — it must never look
/// like a crash.
library;

import 'dart:async';

import 'package:flutter/material.dart';

import '../api/backend.dart';

class HomeScreen extends StatefulWidget {
  final BackendClient client;
  const HomeScreen({super.key, required this.client});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Activity? _activity;
  bool _reachable = false;
  bool _probing = true;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _poll();
    _timer = Timer.periodic(const Duration(seconds: 3), (_) => _poll());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _poll() async {
    try {
      final a = await widget.client.activity();
      if (!mounted) return;
      setState(() {
        _activity = a;
        _reachable = true;
        _probing = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _reachable = false;
        _probing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Vinkona', style: Theme.of(context).textTheme.headlineMedium),
        const SizedBox(height: 16),
        Card(
          child: ListTile(
            leading: _probing
                ? const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : Icon(
                    _reachable ? Icons.check_circle : Icons.cloud_off,
                    color: _reachable ? Colors.green : cs.error,
                  ),
            title: Text(_probing
                ? 'Looking for her…'
                : _reachable
                    ? 'Connected'
                    : 'Not running'),
            subtitle: Text(_probing
                ? widget.client.baseUrl
                : _reachable
                    ? 'The assistant is up on this computer.'
                    : 'Her services aren\'t answering at '
                        '${widget.client.baseUrl}.  Start them with '
                        './vinkona.sh, then this screen will find her.'),
          ),
        ),
        if (_reachable && _activity != null)
          Card(
            child: ListTile(
              leading: Icon(
                _activity!.sessionActive
                    ? Icons.record_voice_over
                    : _activity!.doing == 'idle'
                        ? Icons.nightlight_outlined
                        : Icons.auto_awesome,
                color: cs.primary,
              ),
              title: Text(_activity!.headline),
              subtitle: Text(_activity!.sessionActive
                  ? 'She\'s with you right now.'
                  : _activity!.interruptible
                      ? 'Background work — talking to her takes priority '
                          'automatically.'
                      : 'Finishing something up.'),
            ),
          ),
      ],
    );
  }
}
