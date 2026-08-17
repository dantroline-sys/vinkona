// Profiles: switching asks first, then posts /api/profiles/switch.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:vinkona_desk/api/backend.dart';
import 'package:vinkona_desk/screens/profiles.dart';

void main() {
  testWidgets('switching a profile confirms, then posts the switch',
      (tester) async {
    Map<String, dynamic>? switched;
    final client = BackendClient(
      httpClient: MockClient((req) async {
        if (req.url.path == '/api/profiles/switch') {
          switched = jsonDecode(req.body) as Map<String, dynamic>;
          return http.Response('{"ok": true, "active": "guest"}', 200);
        }
        if (req.url.path == '/api/profiles') {
          return http.Response(
              jsonEncode({
                'active': 'default',
                'profiles': [
                  {
                    'name': 'default',
                    'memories': 42,
                    'personas': 3,
                    'size': 2048,
                    'active': true
                  },
                  {
                    'name': 'guest',
                    'memories': -1,
                    'personas': 1,
                    'size': 0,
                    'active': false
                  },
                ],
              }),
              200);
        }
        return http.Response('{}', 200);
      }),
    );

    await tester.pumpWidget(
        MaterialApp(home: Scaffold(body: ProfilesScreen(client: client))));
    await tester.pumpAndSettle();

    expect(find.text('default'), findsOneWidget);
    expect(find.textContaining('42 memories'), findsOneWidget);

    await tester.tap(find.widgetWithText(TextButton, 'Switch'));
    await tester.pumpAndSettle();
    expect(find.text("Switch to 'guest'?"), findsOneWidget);
    expect(switched, isNull, reason: 'nothing posts before the confirm');

    await tester.tap(find.widgetWithText(FilledButton, 'Switch'));
    await tester.pumpAndSettle();
    expect(switched, isNotNull);
    expect(switched!['name'], 'guest');
  });
}
