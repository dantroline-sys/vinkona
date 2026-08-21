// Local tools against a mock backend: the genre cards render from tools.local,
// Save posts the WHOLE config with parsed feed/account structures, and the
// Test button surfaces the probe verdict.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:vinkona_desk/api/backend.dart';
import 'package:vinkona_desk/screens/local_tools.dart';

BackendClient _client(
    {required Map<String, dynamic> config,
    void Function(Map<String, dynamic>)? onSave,
    Map<String, dynamic>? testReply}) {
  return BackendClient(
    httpClient: MockClient((req) async {
      if (req.url.path == '/api/config' && req.method == 'GET') {
        return http.Response(jsonEncode(config), 200);
      }
      if (req.url.path == '/api/config' && req.method == 'POST') {
        onSave?.call(jsonDecode(req.body) as Map<String, dynamic>);
        return http.Response('{"ok": true}', 200);
      }
      if (req.url.path == '/api/local_tools/test') {
        return http.Response(
            jsonEncode(testReply ?? {'ok': true, 'detail': 'all good'}), 200);
      }
      return http.Response('{}', 200);
    }),
  );
}

const _cfg = {
  'tools': {
    'local': {
      'enabled': true,
      'files': {
        'enabled': true,
        'roots': ['~/Documents']
      },
      'news': {
        'enabled': false,
        'feeds': [
          {'url': 'https://x/rss', 'source': 'X', 'category': 'general'}
        ],
        'poll_interval_s': 1800,
      },
      'weather': {'enabled': true, 'location': 'Hobart'},
      'research': {'enabled': false},
      'mail': {
        'enabled': true,
        'accounts': [
          {
            'label': 'personal',
            'host': 'imap.example.com',
            'port': 993,
            'user': 'me@example.com',
            'password': 'secret'
          }
        ],
      },
      'calendar': {
        'enabled': false,
        'caldav_url': '',
        'user': '',
        'password': '',
        'vinkona_calendar': 'Vinkona'
      },
    }
  }
};

Future<void> _pump(WidgetTester tester, BackendClient client) async {
  // Tall viewport so every genre card builds (the ListView is lazy).
  tester.view.physicalSize = const Size(1200, 3200);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
      home: Scaffold(body: LocalToolsScreen(client: client))));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('renders the genre cards from tools.local', (tester) async {
    await _pump(tester, _client(config: Map.of(_cfg)));
    expect(find.text('Built-in tools on'), findsOneWidget);
    expect(find.text('Files'), findsOneWidget);
    expect(find.text('~/Documents'), findsOneWidget);
    expect(find.text('https://x/rss | X | general'), findsOneWidget);
    expect(find.text('Hobart'), findsOneWidget);
    expect(find.text('imap.example.com'), findsOneWidget);
    // the password fields render obscured
    final pw = tester.widget<TextField>(
        find.widgetWithText(TextField, 'App password').first);
    expect(pw.obscureText, isTrue);
  });

  testWidgets('Save posts the whole config with parsed structures',
      (tester) async {
    Map<String, dynamic>? saved;
    await _pump(tester,
        _client(config: jsonDecode(jsonEncode(_cfg)), onSave: (d) => saved = d));

    // add a second feed line and change the weather town
    await tester.enterText(
        find.widgetWithText(TextField,
            'https://x/rss | X | general'),
        'https://x/rss | X | general\nhttps://y/atom | Y | space');
    await tester.enterText(
        find.widgetWithText(TextField, 'Hobart'), 'Cygnet');
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();

    expect(saved, isNotNull);
    final lt = ((saved!['tools'] as Map)['local'] as Map);
    final feeds = ((lt['news'] as Map)['feeds'] as List).cast<Map>();
    expect(feeds.length, 2);
    expect(feeds[1]['url'], 'https://y/atom');
    expect(feeds[1]['source'], 'Y');
    expect(feeds[1]['category'], 'space');
    expect(((lt['weather'] as Map)['location']), 'Cygnet');
    final accounts = ((lt['mail'] as Map)['accounts'] as List).cast<Map>();
    expect(accounts.single['host'], 'imap.example.com');
    expect(accounts.single['password'], 'secret'); // survives the round trip
  });

  testWidgets('Test surfaces the probe verdict', (tester) async {
    await _pump(
        tester,
        _client(
            config: Map.of(_cfg),
            testReply: {'ok': false, 'detail': 'no folders configured yet'}));
    await tester.tap(find.text('Test').first);
    await tester.pumpAndSettle();
    expect(find.textContaining('no folders configured yet'), findsOneWidget);
  });
}
