// Everyday options: the Basic-tier knobs that are NOT feature-recipe switches
// and not persona/voice (those have their own surfaces).
//
// The list itself is server-driven: FIELD_LEVELS says which paths are Basic,
// FEATURE_RECIPES says which of those the Features tab already covers, and this
// screen renders what remains — friendly labels and fixed-choice pickers come
// from the same /api/field_levels payload, per-field help from /api/help.  The
// only Dart here is presentation: which section header a path sits under and
// which widget its VALUE TYPE gets (switch for booleans, picker for fixed
// choices, text for the rest).
import 'package:flutter/material.dart';

import '../api/backend.dart';
import '../widgets/load_problem.dart';

/// Paths rendered by the Persona & voice tab, not here.
const _dedicated = {'default_persona', 'tts.engine', 'tts.default_voice'};

class EverydayScreen extends StatefulWidget {
  final BackendClient client;
  const EverydayScreen({super.key, required this.client});

  @override
  State<EverydayScreen> createState() => _EverydayScreenState();
}

class _EverydayScreenState extends State<EverydayScreen> {
  Map<String, dynamic>? _config;
  FieldLevels? _levels;
  Set<String> _recipePaths = const {};
  Map<String, String> _help = const {};
  final Map<String, TextEditingController> _text = {};
  String? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    for (final c in _text.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final cfg = await widget.client.config();
      final levels = await widget.client.fieldLevels();
      final recipes = await widget.client.featureRecipes();
      final help = await widget.client.help();
      if (!mounted) return;
      setState(() {
        _config = cfg;
        _levels = levels;
        _recipePaths = recipes.map((r) => r.path).toSet();
        _help = help;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  void _toast(String msg) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(msg)));

  List<String> _fields() {
    final out = _levels!.levels.entries
        .where((e) => e.value == 'basic')
        .map((e) => e.key)
        .where((p) => !_recipePaths.contains(p) && !_dedicated.contains(p))
        .where((p) {
      final v = configGet(_config!, p);
      return v != null && v is! Map; // leaves only, present in this config
    }).toList();
    out.sort();
    return out;
  }

  // Section headers are presentation: a friendly name per config area, in a
  // deliberate order (the conversational stuff first, machinery last).
  static const _sections = [
    'Conversation',
    'Her sense of the day',
    'Memory & knowledge',
    'Research',
    'Reminders',
    'Tools',
  ];

  String _sectionOf(String path) {
    if (path.startsWith('awareness.')) return 'Her sense of the day';
    if (path.startsWith('memory.') || path.startsWith('knowledge.')) {
      return 'Memory & knowledge';
    }
    if (path.startsWith('research.')) return 'Research';
    if (path.startsWith('notifications.')) return 'Reminders';
    if (path.startsWith('tools.')) return 'Tools';
    return 'Conversation';
  }

  String _labelOf(String path) {
    final friendly = _levels!.labels[path];
    if (friendly != null && friendly.isNotEmpty) return friendly;
    final leaf = path.split('.').last.replaceAll('_', ' ');
    return leaf.isEmpty ? path : leaf[0].toUpperCase() + leaf.substring(1);
  }

  Future<void> _save(String path, Object? value, {String? toast}) async {
    setState(() => _busy = true);
    try {
      configSet(_config!, path, value);
      await widget.client.saveConfig(_config!);
      if (!mounted) return;
      _toast(toast ?? '${_labelOf(path)} saved.');
    } catch (e) {
      if (!mounted) return;
      _toast('Could not save: $e');
      await _load(); // re-sync with what the backend actually has
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _choiceLabel(Object? v) {
    if (v == true) return 'On';
    if (v == false) return 'Off';
    final s = v.toString();
    return s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
  }

  TextEditingController _controllerFor(String path, String initial) =>
      _text.putIfAbsent(path, () => TextEditingController(text: initial));

  /// Parse a comma-separated field back into a list shaped like the original.
  void _saveList(String path, List original, String raw) {
    final parts = raw
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    if (original.every((e) => e is int)) {
      final nums = parts.map(int.tryParse).toList();
      if (nums.contains(null)) {
        _toast('Please use whole numbers, separated by commas.');
        return;
      }
      _save(path, nums.whereType<int>().toList());
    } else {
      _save(path, parts);
    }
  }

  Widget _fieldTile(String path) {
    final value = configGet(_config!, path);
    final label = _labelOf(path);
    final help = _help[path] ?? '';
    final choices = _levels!.choices[path];

    if (choices != null) {
      final selected = choices.indexWhere((c) => c == value);
      return ListTile(
        title: Text(label),
        subtitle: help.isEmpty
            ? null
            : Text(help, maxLines: 2, overflow: TextOverflow.ellipsis),
        trailing: SegmentedButton<int>(
          segments: [
            for (var i = 0; i < choices.length; i++)
              ButtonSegment(value: i, label: Text(_choiceLabel(choices[i]))),
          ],
          selected: {selected < 0 ? 0 : selected},
          onSelectionChanged: _busy
              ? null
              : (sel) => _save(path, choices[sel.first],
                  toast: '$label: ${_choiceLabel(choices[sel.first])}.'),
        ),
      );
    }
    if (value is bool) {
      return SwitchListTile(
        title: Text(label),
        subtitle: help.isEmpty
            ? null
            : Text(help, maxLines: 2, overflow: TextOverflow.ellipsis),
        value: value,
        onChanged: _busy
            ? null
            : (v) =>
                _save(path, v, toast: '$label is ${v ? "on" : "off"}.'),
      );
    }
    if (value is List) {
      final ctl = _controllerFor(path, value.join(', '));
      return ListTile(
        title: Text(label),
        subtitle: help.isEmpty
            ? null
            : Text(help, maxLines: 2, overflow: TextOverflow.ellipsis),
        trailing: SizedBox(
          width: 180,
          child: TextField(
            controller: ctl,
            enabled: !_busy,
            decoration: const InputDecoration(isDense: true),
            onSubmitted: (raw) => _saveList(path, value, raw),
          ),
        ),
      );
    }
    if (value is num) {
      final ctl = _controllerFor(path, '$value');
      return ListTile(
        title: Text(label),
        subtitle: help.isEmpty
            ? null
            : Text(help, maxLines: 2, overflow: TextOverflow.ellipsis),
        trailing: SizedBox(
          width: 100,
          child: TextField(
            controller: ctl,
            enabled: !_busy,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(isDense: true),
            onSubmitted: (raw) {
              final n = value is int ? int.tryParse(raw) : double.tryParse(raw);
              if (n == null) {
                _toast('Please enter a number.');
                return;
              }
              _save(path, n);
            },
          ),
        ),
      );
    }
    // Everything else edits as text (locations, country codes…).
    final ctl = _controllerFor(path, value.toString());
    return ListTile(
      title: Text(label),
      subtitle: help.isEmpty
          ? null
          : Text(help, maxLines: 2, overflow: TextOverflow.ellipsis),
      trailing: SizedBox(
        width: 220,
        child: TextField(
          controller: ctl,
          enabled: !_busy,
          decoration: const InputDecoration(isDense: true),
          onSubmitted: (raw) => _save(path, raw),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return LoadProblem(detail: _error!, onRetry: _load);
    }
    if (_config == null || _levels == null) {
      return const Center(child: CircularProgressIndicator());
    }
    final bySection = <String, List<String>>{};
    for (final p in _fields()) {
      bySection.putIfAbsent(_sectionOf(p), () => []).add(p);
    }
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Everyday options',
            style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        Text('Type-in fields save when you press Enter.',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 8),
        for (final section in _sections)
          if (bySection.containsKey(section)) ...[
            const SizedBox(height: 16),
            Text(section, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Card(
              child: Column(children: [
                for (final p in bySection[section]!) _fieldTile(p),
              ]),
            ),
          ],
      ],
    );
  }
}
