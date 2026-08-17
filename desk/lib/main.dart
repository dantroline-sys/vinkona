/// Vinkona's desktop face.
///
/// A thin client over the backend's localhost API (see lib/api/backend.dart for
/// the one rule: no logic in Dart).  The design brief, from Dan: extremely easy
/// for a non-technical person — a handful of plain-language switches with
/// preview-then-confirm — with the advanced machinery hidden but reachable.
library;

import 'package:flutter/material.dart';

import 'api/backend.dart';
import 'screens/home.dart';
import 'screens/placeholder.dart';
import 'screens/settings_basic.dart';

void main() {
  runApp(VinkonaDeskApp(client: BackendClient()));
}

class VinkonaDeskApp extends StatelessWidget {
  final BackendClient client;
  const VinkonaDeskApp({super.key, required this.client});

  @override
  Widget build(BuildContext context) {
    // A calm, warm look — a companion's window, not an admin console.  One
    // seeded hue keeps light and dark coherent without hand-tuning either.
    const seed = Color(0xFF6D5BA6); // muted violet
    return MaterialApp(
      title: 'Vinkona',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: seed),
        visualDensity: VisualDensity.comfortable,
      ),
      darkTheme: ThemeData(
        colorScheme:
            ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark),
        visualDensity: VisualDensity.comfortable,
      ),
      home: Shell(client: client),
    );
  }
}

class Shell extends StatefulWidget {
  final BackendClient client;
  const Shell({super.key, required this.client});

  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> {
  int _index = 0;

  @override
  Widget build(BuildContext context) {
    final pages = <Widget>[
      HomeScreen(client: widget.client),
      SettingsBasicScreen(client: widget.client),
      const ComingSoonScreen(
          title: 'Her tools',
          detail: 'The tools she makes for herself — the queue of ideas, the '
              'builds, and the ones that need your eye.  (Stage D4 of the '
              'desktop plan.)'),
      const ComingSoonScreen(
          title: 'Live',
          detail: 'Watch what the language models are seeing and saying, as it '
              'happens.  (Stage D4 of the desktop plan.)'),
    ];
    return Scaffold(
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: _index,
            onDestinationSelected: (i) => setState(() => _index = i),
            labelType: NavigationRailLabelType.all,
            leading: const Padding(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: CircleAvatar(child: Text('V')),
            ),
            destinations: const [
              NavigationRailDestination(
                  icon: Icon(Icons.home_outlined),
                  selectedIcon: Icon(Icons.home),
                  label: Text('Home')),
              NavigationRailDestination(
                  icon: Icon(Icons.tune_outlined),
                  selectedIcon: Icon(Icons.tune),
                  label: Text('Settings')),
              NavigationRailDestination(
                  icon: Icon(Icons.handyman_outlined),
                  selectedIcon: Icon(Icons.handyman),
                  label: Text('Tools')),
              NavigationRailDestination(
                  icon: Icon(Icons.monitor_heart_outlined),
                  selectedIcon: Icon(Icons.monitor_heart),
                  label: Text('Live')),
            ],
          ),
          const VerticalDivider(width: 1),
          Expanded(child: pages[_index]),
        ],
      ),
    );
  }
}
