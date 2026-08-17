// Persona & voice: who she is and how she sounds.
//
// Personas come from the backend's personas document and are chosen by saving
// the document back whole with a new `default` (the web panel's contract).
// The speech engine comes from /api/tts (with install state per engine) and is
// switched via /api/tts/select — the backend owns what a switch entails
// (provisioning, which services restart); this screen only asks "now or at the
// next restart?".  The spoken voice is the tts.default_voice config knob,
// offered as a picker exactly when the current engine has preset voices.
import 'package:flutter/material.dart';

import '../api/backend.dart';
import '../widgets/load_problem.dart';

class PersonaVoiceScreen extends StatefulWidget {
  final BackendClient client;
  const PersonaVoiceScreen({super.key, required this.client});

  @override
  State<PersonaVoiceScreen> createState() => _PersonaVoiceScreenState();
}

class _PersonaVoiceScreenState extends State<PersonaVoiceScreen> {
  PersonaDoc? _doc;
  TtsStatus? _tts;
  Map<String, dynamic>? _config;
  Map<String, String> _help = const {};
  String? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final doc = await widget.client.personas();
      final tts = await widget.client.tts();
      final cfg = await widget.client.config();
      final help = await widget.client.help();
      if (!mounted) return;
      setState(() {
        _doc = doc;
        _tts = tts;
        _config = cfg;
        _help = help;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  void _toast(String msg) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(msg)));

  String _pretty(String name) =>
      name.isEmpty ? name : name[0].toUpperCase() + name.substring(1);

  Future<void> _choosePersona(Persona p) async {
    if (_doc == null || p.name == _doc!.defaultName) return;
    setState(() => _busy = true);
    try {
      final doc = Map<String, dynamic>.from(_doc!.raw);
      doc['default'] = p.name;
      await widget.client.savePersonas(doc);
      if (!mounted) return;
      _toast("She'll be ${_pretty(p.name)} from the next conversation.");
      await _load();
    } catch (e) {
      if (!mounted) return;
      _toast('Could not switch persona: $e');
      await _load();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// The engine confirm sheet: what the engine is, whether it still needs
  /// installing, and when to apply — the backend does the rest.
  Future<void> _chooseEngine(TtsEngine e) async {
    if (_tts == null || e.current) return;
    final when = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      builder: (ctx) => SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Switch her voice to ${e.label}',
                style: Theme.of(ctx).textTheme.titleLarge),
            const SizedBox(height: 12),
            Text(e.note),
            const SizedBox(height: 8),
            Text(e.footprint, style: Theme.of(ctx).textTheme.bodySmall),
            if (!e.installed) ...[
              const SizedBox(height: 8),
              Text(
                  "It isn't set up yet — it will be installed automatically "
                  'when she restarts.',
                  style: Theme.of(ctx)
                      .textTheme
                      .bodySmall
                      ?.copyWith(fontStyle: FontStyle.italic)),
            ],
            const SizedBox(height: 20),
            // OverflowBar: the three buttons stack when the sheet is narrow.
            OverflowBar(
              alignment: MainAxisAlignment.end,
              spacing: 8,
              overflowSpacing: 8,
              children: [
                TextButton(
                    onPressed: () => Navigator.pop(ctx),
                    child: const Text('Cancel')),
                OutlinedButton(
                    onPressed: () => Navigator.pop(ctx, 'next_restart'),
                    child: const Text('At the next restart')),
                FilledButton(
                    onPressed: () => Navigator.pop(ctx, 'now'),
                    child: const Text('Restart her now')),
              ],
            ),
          ],
        ),
      ),
    );
    if (when == null) return;
    setState(() => _busy = true);
    try {
      final res = await widget.client.ttsSelect(e.key, when: when);
      if (!mounted) return;
      _toast(res['restarting'] == true
          ? "Switching — she's restarting with ${e.label}."
          : "Done — she'll speak with ${e.label} after the next restart.");
      await _load();
    } catch (err) {
      if (!mounted) return;
      _toast('Could not switch the engine: $err');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _chooseVoice(String voice) async {
    if (_config == null) return;
    setState(() => _busy = true);
    try {
      configSet(_config!, 'tts.default_voice', voice);
      await widget.client.saveConfig(_config!);
      if (!mounted) return;
      _toast("Her voice is now '$voice'.");
      await _load();
    } catch (e) {
      if (!mounted) return;
      _toast('Could not save the voice: $e');
      await _load();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return LoadProblem(detail: _error!, onRetry: _load);
    }
    if (_doc == null || _tts == null) {
      return const Center(child: CircularProgressIndicator());
    }
    final voices = _tts!.currentEngine?.voices ?? const <String>[];
    final currentVoice = (configGet(_config!, 'tts.default_voice') ?? '')
        .toString();
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Who she is', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        Text('Pick the personality she greets you with.',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 12),
        for (final p in _doc!.personas)
          Card(
            child: ListTile(
              leading: Icon(
                p.name == _doc!.defaultName
                    ? Icons.check_circle
                    : Icons.circle_outlined,
                color: p.name == _doc!.defaultName
                    ? Theme.of(context).colorScheme.primary
                    : Theme.of(context).colorScheme.outline,
              ),
              title: Text(_pretty(p.name)),
              subtitle: Text(
                [
                  if (p.description.isNotEmpty) p.description,
                  if (p.voice.isNotEmpty) 'Speaks as ${p.voice}.',
                ].join('  '),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              onTap: _busy ? null : () => _choosePersona(p),
            ),
          ),
        const SizedBox(height: 24),
        Text('How she speaks',
            style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        Text('The speech engine — each has its own character and footprint.',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 12),
        for (final e in _tts!.engines)
          Card(
            child: ListTile(
              leading: Icon(
                e.current ? Icons.check_circle : Icons.circle_outlined,
                color: e.current
                    ? Theme.of(context).colorScheme.primary
                    : Theme.of(context).colorScheme.outline,
              ),
              title: Row(children: [
                Text(e.label),
                if (!e.installed) ...[
                  const SizedBox(width: 8),
                  Chip(
                    label: const Text('not installed yet'),
                    visualDensity: VisualDensity.compact,
                    labelStyle: Theme.of(context).textTheme.labelSmall,
                  ),
                ],
              ]),
              subtitle:
                  Text(e.note, maxLines: 2, overflow: TextOverflow.ellipsis),
              onTap: _busy ? null : () => _chooseEngine(e),
            ),
          ),
        if (voices.isNotEmpty) ...[
          const SizedBox(height: 24),
          Text('Her voice', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 4),
          Text(
              _help['tts.default_voice'] ??
                  'The preset she speaks with by default.',
              style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final v in voices)
                ChoiceChip(
                  label: Text(v),
                  selected: v == currentVoice,
                  onSelected: _busy
                      ? null
                      : (sel) {
                          if (sel) _chooseVoice(v);
                        },
                ),
            ],
          ),
        ],
      ],
    );
  }
}
