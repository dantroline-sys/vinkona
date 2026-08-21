// Local tools: the bundled toolset (VIN-LOCAL-01) — calendar, mail, files,
// news, weather and research served from the machine Vinkona runs on.
//
// A thin, purpose-built surface over tools.local: the Everyday tab's generic
// renderer can't do the structured parts (feed rows, mail accounts) or the
// per-genre Test probes, so they live here.  Same contract as everywhere:
// load the WHOLE config, edit in place, save it back whole; Test runs one
// real probe against the SAVED settings.
import 'package:flutter/material.dart';

import '../api/backend.dart';
import '../widgets/load_problem.dart';

class LocalToolsScreen extends StatefulWidget {
  final BackendClient client;
  const LocalToolsScreen({super.key, required this.client});

  @override
  State<LocalToolsScreen> createState() => _LocalToolsScreenState();
}

class _MailAccount {
  final label = TextEditingController();
  final host = TextEditingController();
  final port = TextEditingController(text: '993');
  final user = TextEditingController();
  final password = TextEditingController();

  _MailAccount([Map<String, dynamic>? a]) {
    if (a != null) {
      label.text = '${a['label'] ?? ''}';
      host.text = '${a['host'] ?? ''}';
      port.text = '${a['port'] ?? 993}';
      user.text = '${a['user'] ?? ''}';
      password.text = '${a['password'] ?? ''}';
    }
  }

  bool get filled =>
      host.text.trim().isNotEmpty || user.text.trim().isNotEmpty;

  Map<String, dynamic> toJson() => {
        'label': label.text.trim().isEmpty
            ? (user.text.trim().isEmpty ? 'mail' : user.text.trim())
            : label.text.trim(),
        'host': host.text.trim(),
        'port': int.tryParse(port.text.trim()) ?? 993,
        'user': user.text.trim(),
        'password': password.text,
      };

  void dispose() {
    for (final c in [label, host, port, user, password]) {
      c.dispose();
    }
  }
}

class _LocalToolsScreenState extends State<LocalToolsScreen> {
  Map<String, dynamic>? _config;
  String? _error;
  bool _busy = false;

  bool _master = false;
  final _on = <String, bool>{};                 // genre → enabled
  final _filesRoots = TextEditingController();
  final _newsFeeds = TextEditingController();
  final _newsInterval = TextEditingController(text: '1800');
  final _weatherLoc = TextEditingController();
  final _calUrl = TextEditingController();
  final _calUser = TextEditingController();
  final _calPass = TextEditingController();
  final _calOwn = TextEditingController(text: 'Vinkona');
  final List<_MailAccount> _accounts = [];
  final _testResult = <String, String>{};       // genre → last probe verdict
  final _testOk = <String, bool>{};

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    for (final c in [_filesRoots, _newsFeeds, _newsInterval, _weatherLoc,
                     _calUrl, _calUser, _calPass, _calOwn]) {
      c.dispose();
    }
    for (final a in _accounts) {
      a.dispose();
    }
    super.dispose();
  }

  Map<String, dynamic> _genre(Map<String, dynamic> lt, String name) =>
      (lt[name] is Map<String, dynamic>)
          ? lt[name] as Map<String, dynamic>
          : <String, dynamic>{};

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final cfg = await widget.client.config();
      if (!mounted) return;
      final lt = _genre(
          (cfg['tools'] is Map<String, dynamic>)
              ? cfg['tools'] as Map<String, dynamic>
              : <String, dynamic>{},
          'local');
      setState(() {
        _config = cfg;
        _master = lt['enabled'] == true;
        for (final g in ['files', 'news', 'weather', 'research', 'mail', 'calendar']) {
          _on[g] = _genre(lt, g)['enabled'] == true;
        }
        _filesRoots.text =
            ((_genre(lt, 'files')['roots'] as List?) ?? []).join('\n');
        _newsFeeds.text = ((_genre(lt, 'news')['feeds'] as List?) ?? [])
            .map((f) => f is Map
                ? [f['url'] ?? '', f['source'] ?? '', f['category'] ?? '']
                    .join(' | ')
                    .replaceAll(RegExp(r'( \| )+$'), '')
                : '$f')
            .join('\n');
        _newsInterval.text = '${_genre(lt, 'news')['poll_interval_s'] ?? 1800}';
        _weatherLoc.text = '${_genre(lt, 'weather')['location'] ?? ''}';
        _calUrl.text = '${_genre(lt, 'calendar')['caldav_url'] ?? ''}';
        _calUser.text = '${_genre(lt, 'calendar')['user'] ?? ''}';
        _calPass.text = '${_genre(lt, 'calendar')['password'] ?? ''}';
        _calOwn.text =
            '${_genre(lt, 'calendar')['vinkona_calendar'] ?? 'Vinkona'}';
        for (final a in _accounts) {
          a.dispose();
        }
        _accounts
          ..clear()
          ..addAll(((_genre(lt, 'mail')['accounts'] as List?) ?? [])
              .whereType<Map>()
              .map((a) => _MailAccount(a.cast<String, dynamic>())));
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  void _toast(String msg) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(msg)));

  Future<void> _save() async {
    if (_config == null) return;
    setState(() => _busy = true);
    try {
      final feeds = _newsFeeds.text
          .split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .map((l) {
        final p = l.split('|').map((x) => x.trim()).toList();
        return {
          'url': p.isNotEmpty ? p[0] : '',
          'source': p.length > 1 ? p[1] : '',
          'category': p.length > 2 ? p[2] : '',
        };
      }).where((f) => (f['url'] as String).isNotEmpty).toList();
      configSet(_config!, 'tools.local.enabled', _master);
      configSet(_config!, 'tools.local.files.enabled', _on['files']);
      configSet(
          _config!,
          'tools.local.files.roots',
          _filesRoots.text
              .split('\n')
              .map((s) => s.trim())
              .where((s) => s.isNotEmpty)
              .toList());
      configSet(_config!, 'tools.local.news.enabled', _on['news']);
      configSet(_config!, 'tools.local.news.feeds', feeds);
      configSet(_config!, 'tools.local.news.poll_interval_s',
          int.tryParse(_newsInterval.text.trim()) ?? 1800);
      configSet(_config!, 'tools.local.weather.enabled', _on['weather']);
      configSet(
          _config!, 'tools.local.weather.location', _weatherLoc.text.trim());
      configSet(_config!, 'tools.local.research.enabled', _on['research']);
      configSet(_config!, 'tools.local.mail.enabled', _on['mail']);
      configSet(_config!, 'tools.local.mail.accounts',
          _accounts.where((a) => a.filled).map((a) => a.toJson()).toList());
      configSet(_config!, 'tools.local.calendar.enabled', _on['calendar']);
      configSet(
          _config!, 'tools.local.calendar.caldav_url', _calUrl.text.trim());
      configSet(_config!, 'tools.local.calendar.user', _calUser.text.trim());
      configSet(_config!, 'tools.local.calendar.password', _calPass.text);
      configSet(
          _config!,
          'tools.local.calendar.vinkona_calendar',
          _calOwn.text.trim().isEmpty ? 'Vinkona' : _calOwn.text.trim());
      await widget.client.saveConfig(_config!);
      if (!mounted) return;
      _toast('Saved — a new chat picks it up; background work after a restart.');
    } catch (e) {
      if (!mounted) return;
      _toast('Could not save: $e');
      await _load();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _test(String genre) async {
    setState(() {
      _testResult[genre] = 'testing…';
      _testOk[genre] = true;
    });
    Map<String, dynamic> d;
    try {
      d = await widget.client.localToolsTest(genre);
    } catch (e) {
      d = {'ok': false, 'detail': '$e'};
    }
    if (!mounted) return;
    setState(() {
      _testOk[genre] = d['ok'] == true;
      _testResult[genre] =
          '${d['ok'] == true ? '✓' : '✗'} ${d['detail'] ?? ''}';
    });
  }

  Widget _testRow(String genre) {
    final res = _testResult[genre];
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        OutlinedButton(
          onPressed: _busy ? null : () => _test(genre),
          child: const Text('Test'),
        ),
        const SizedBox(width: 12),
        if (res != null)
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(res,
                  style: TextStyle(
                      fontSize: 12,
                      color: (_testOk[genre] ?? true)
                          ? null
                          : Theme.of(context).colorScheme.error)),
            ),
          ),
      ],
    );
  }

  Widget _card(String genre, String title, String subtitle,
      List<Widget> children) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: Text(title),
              subtitle: Text(subtitle,
                  maxLines: 2, overflow: TextOverflow.ellipsis),
              value: _on[genre] ?? false,
              onChanged:
                  _busy ? null : (v) => setState(() => _on[genre] = v),
            ),
            ...children,
            const SizedBox(height: 8),
            _testRow(genre),
          ],
        ),
      ),
    );
  }

  Widget _text(TextEditingController ctl, String label,
      {String? hint, int lines = 1, bool obscure = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: TextField(
        controller: ctl,
        enabled: !_busy,
        maxLines: obscure ? 1 : lines,
        obscureText: obscure,
        autocorrect: false,
        decoration: InputDecoration(
            labelText: label, hintText: hint, isDense: true),
      ),
    );
  }

  Widget _accountEditor(int i) {
    final a = _accounts[i];
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(children: [
        Row(children: [
          Expanded(child: _text(a.label, 'Label', hint: 'personal')),
          const SizedBox(width: 8),
          Expanded(
              flex: 2,
              child: _text(a.host, 'IMAP server', hint: 'e.g. imap.example.com')),
          const SizedBox(width: 8),
          SizedBox(width: 70, child: _text(a.port, 'Port')),
          IconButton(
            tooltip: 'Remove account',
            icon: const Icon(Icons.close),
            onPressed: _busy
                ? null
                : () => setState(() => _accounts.removeAt(i).dispose()),
          ),
        ]),
        Row(children: [
          Expanded(child: _text(a.user, 'Username', hint: 'you@example.com')),
          const SizedBox(width: 8),
          Expanded(child: _text(a.password, 'App password', obscure: true)),
        ]),
      ]),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return LoadProblem(detail: _error!, onRetry: _load);
    }
    if (_config == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Local tools', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        Text(
            'Calendar, mail, files, news, weather and research — served from this '
            'machine, no Mac needed. Each is its own switch and stays off until '
            'configured. Mail is read-only; calendar writes only ever land on her '
            'own calendar. Save first, then Test runs a real probe.',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 12),
        Card(
          child: SwitchListTile(
            title: const Text('Built-in tools on'),
            subtitle: const Text(
                'The master switch — nothing below works while this is off.'),
            value: _master,
            onChanged: _busy ? null : (v) => setState(() => _master = v),
          ),
        ),
        const SizedBox(height: 8),
        _card('files', 'Files', 'She may search & read ONLY these folders.', [
          _text(_filesRoots, 'Folders she may read (one per line)',
              hint: 'e.g. ~/Documents', lines: 3),
        ]),
        _card('news', 'News feeds',
            'Polled into her durable archive; headlines & event memory serve from it.', [
          _text(_newsFeeds, 'Feeds (address | source | category, one per line)',
              hint: 'https://feeds.bbci.co.uk/news/rss.xml | BBC | general',
              lines: 4),
          _text(_newsInterval, 'Check every (seconds)'),
        ]),
        _card('weather', 'Weather', 'Keyless Open-Meteo forecasts.', [
          _text(_weatherLoc, 'Your town or city', hint: 'e.g. Hobart'),
        ]),
        _card(
            'research',
            'Research lookups',
            'Europe PMC, OpenAlex, Wikipedia, Stack Exchange and friends — '
                'keyless, from this machine, every call audited.',
            const []),
        _card('mail', 'Mail', 'IMAP, strictly read-only — use an app password.', [
          for (var i = 0; i < _accounts.length; i++) _accountEditor(i),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: _busy
                  ? null
                  : () => setState(() => _accounts.add(_MailAccount())),
              icon: const Icon(Icons.add),
              label: const Text('Add account'),
            ),
          ),
        ]),
        _card('calendar', 'Calendar',
            'CalDAV — reads span all calendars; writes go to ONE calendar of hers.', [
          _text(_calUrl, 'Server address',
              hint: 'https://caldav.icloud.com or your Nextcloud URL'),
          _text(_calUser, 'Username'),
          _text(_calPass, 'App password', obscure: true),
          _text(_calOwn, 'The calendar she may write to (create it first)'),
        ]),
        const SizedBox(height: 12),
        Row(children: [
          FilledButton(
            onPressed: _busy ? null : _save,
            child: const Text('Save'),
          ),
          const SizedBox(width: 12),
          if (_busy)
            const SizedBox(
                width: 18, height: 18,
                child: CircularProgressIndicator(strokeWidth: 2)),
        ]),
        const SizedBox(height: 24),
      ],
    );
  }
}
