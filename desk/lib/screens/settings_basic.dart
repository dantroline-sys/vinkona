/// Settings, the Basic tier: a short list of plain-language feature switches.
///
/// Everything here is driven by the backend: FIELD_LEVELS says which few knobs
/// a non-technical person should see, FEATURE_RECIPES supplies the
/// preview-then-confirm story for each switch (what turning it on really does,
/// and the companion settings that follow).  This screen renders those; it
/// decides nothing itself.
///
/// Advanced/expert settings stay hidden behind one deliberate gate (stage D4
/// renders the full tree; until then the gate points at the web panel).
library;

import 'package:flutter/material.dart';

import '../api/backend.dart';

class SettingsBasicScreen extends StatefulWidget {
  final BackendClient client;
  const SettingsBasicScreen({super.key, required this.client});

  @override
  State<SettingsBasicScreen> createState() => _SettingsBasicScreenState();
}

class _SettingsBasicScreenState extends State<SettingsBasicScreen> {
  Map<String, dynamic>? _config;
  List<FeatureRecipe> _recipes = const [];
  String? _error;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final cfg = await widget.client.config();
      final recipes = await widget.client.featureRecipes();
      if (!mounted) return;
      setState(() {
        _config = cfg;
        _recipes = recipes;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Could not load her settings: $e');
    }
  }

  bool _isOn(FeatureRecipe r) => configGet(_config!, r.path) == true;

  /// The preview-then-confirm sheet: say what the flip means in plain words,
  /// list the companion changes, apply only on confirm.
  Future<void> _confirmFlip(FeatureRecipe r, bool turnOn) async {
    final side = turnOn ? r.enable : r.disable;
    final ok = await showModalBottomSheet<bool>(
      context: context,
      showDragHandle: true,
      builder: (ctx) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${turnOn ? "Turn on" : "Turn off"}: ${r.title}',
                style: Theme.of(ctx).textTheme.titleLarge),
            const SizedBox(height: 12),
            Text(side.summary),
            if (side.changes.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text('This also changes:',
                  style: Theme.of(ctx).textTheme.labelLarge),
              for (final c in side.changes)
                Padding(
                  padding: const EdgeInsets.only(left: 8, top: 4),
                  child: Row(children: [
                    const Icon(Icons.subdirectory_arrow_right, size: 16),
                    const SizedBox(width: 6),
                    Expanded(child: Text(c.label)),
                  ]),
                ),
            ],
            if (side.note.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(side.note,
                  style: Theme.of(ctx)
                      .textTheme
                      .bodySmall
                      ?.copyWith(fontStyle: FontStyle.italic)),
            ],
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                    onPressed: () => Navigator.pop(ctx, false),
                    child: const Text('Cancel')),
                const SizedBox(width: 8),
                FilledButton(
                    onPressed: () => Navigator.pop(ctx, true),
                    child: Text(turnOn ? 'Turn on' : 'Turn off')),
              ],
            ),
          ],
        ),
      ),
    );
    if (ok != true || _config == null) return;

    // Apply the switch and its companion changes, then save the whole config —
    // the same contract the web panel uses.
    configSet(_config!, r.path, turnOn);
    for (final c in side.changes) {
      configSet(_config!, c.path, c.value);
    }
    setState(() => _saving = true);
    try {
      await widget.client.saveConfig(_config!);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content:
              Text('${r.title} ${turnOn ? "is on" : "is off"}.')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Could not save: $e')));
      await _load(); // re-sync with what the backend actually has
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(_error!),
          const SizedBox(height: 12),
          FilledButton(onPressed: _load, child: const Text('Try again')),
        ]),
      );
    }
    if (_config == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('What she may do',
            style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        Text('Each switch explains itself before anything changes.',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 16),
        for (final r in _recipes)
          Card(
            child: SwitchListTile(
              title: Text(r.title),
              subtitle: Text(
                _isOn(r) ? r.disable.summary : r.enable.summary,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              value: _isOn(r),
              onChanged: _saving ? null : (v) => _confirmFlip(r, v),
            ),
          ),
        const SizedBox(height: 24),
        // The one deliberate gate to the machinery.
        OutlinedButton.icon(
          icon: const Icon(Icons.settings_suggest_outlined),
          label: const Text('Advanced settings'),
          onPressed: () => showDialog<void>(
            context: context,
            builder: (ctx) => AlertDialog(
              title: const Text('Advanced settings'),
              content: const Text(
                  'The full settings tree lands here in stage D4.  Until then '
                  'the web panel has everything: open '
                  'http://127.0.0.1:8090 in a browser on this computer.'),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(ctx),
                    child: const Text('OK')),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
