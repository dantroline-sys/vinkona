// Profiles: whole memory+personas bundles ("who is she talking to").
// Switching restarts the services that hold the memory DB — the backend does
// that; this screen asks first and reports after.
import 'package:flutter/material.dart';

import '../api/backend.dart';
import '../widgets/load_problem.dart';

class ProfilesScreen extends StatefulWidget {
  final BackendClient client;
  const ProfilesScreen({super.key, required this.client});

  @override
  State<ProfilesScreen> createState() => _ProfilesScreenState();
}

class _ProfilesScreenState extends State<ProfilesScreen> {
  ProfilesStatus? _status;
  String? _error;
  bool _busy = false;
  final _newName = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _newName.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final s = await widget.client.profiles();
      if (!mounted) return;
      setState(() => _status = s);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  void _toast(String msg) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(msg)));

  String _size(int bytes) {
    if (bytes <= 0) return 'empty';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).round()} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  Future<void> _switchTo(ProfileInfo p) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text("Switch to '${p.name}'?"),
        content: const Text(
            'Her memories and personas swap to this bundle, and she restarts '
            'to pick it up.  Nothing is deleted — switching back restores '
            'everything.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Switch')),
        ],
      ),
    );
    if (ok != true) return;
    setState(() => _busy = true);
    try {
      await widget.client.profileSwitch(p.name);
      if (!mounted) return;
      _toast("Switched — she's now on '${p.name}'.");
      await _load();
    } catch (e) {
      if (!mounted) return;
      _toast('Could not switch: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _create() async {
    final name = _newName.text.trim();
    if (name.isEmpty) return;
    setState(() => _busy = true);
    try {
      await widget.client.profileCreate(name);
      if (!mounted) return;
      _newName.clear();
      _toast("Created '$name' — switch to it when you're ready.");
      await _load();
    } catch (e) {
      if (!mounted) return;
      _toast('Could not create it: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return LoadProblem(detail: _error!, onRetry: _load);
    }
    if (_status == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Profiles', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        Text(
            'Separate memory bundles — one per person she talks to, or one '
            'per household role.',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 16),
        for (final p in _status!.profiles)
          Card(
            child: ListTile(
              leading: Icon(
                p.active ? Icons.check_circle : Icons.circle_outlined,
                color: p.active
                    ? Theme.of(context).colorScheme.primary
                    : Theme.of(context).colorScheme.outline,
              ),
              title: Text(p.name),
              subtitle: Text([
                p.memories < 0 ? 'new' : '${p.memories} memories',
                '${p.personas} personas',
                _size(p.sizeBytes),
              ].join(' · ')),
              trailing: p.active
                  ? Text('active',
                      style: Theme.of(context).textTheme.labelMedium)
                  : TextButton(
                      onPressed: _busy ? null : () => _switchTo(p),
                      child: const Text('Switch')),
            ),
          ),
        const SizedBox(height: 16),
        Row(children: [
          Expanded(
            child: TextField(
              controller: _newName,
              enabled: !_busy,
              decoration: const InputDecoration(
                labelText: 'New profile name',
                isDense: true,
              ),
              onSubmitted: (_) => _create(),
            ),
          ),
          const SizedBox(width: 12),
          FilledButton(
              onPressed: _busy ? null : _create, child: const Text('Create')),
        ]),
      ],
    );
  }
}
