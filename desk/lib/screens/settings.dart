// Settings, assembled: four Basic-tier surfaces under one roof.
// Persona & voice and Features lead (the things a person actually changes);
// Everyday holds the remaining Basic knobs; Profiles the memory bundles.
import 'package:flutter/material.dart';

import '../api/backend.dart';
import 'everyday.dart';
import 'persona_voice.dart';
import 'profiles.dart';
import 'settings_basic.dart';

class SettingsScreen extends StatelessWidget {
  final BackendClient client;
  const SettingsScreen({super.key, required this.client});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 4,
      child: Column(
        children: [
          Material(
            color: Theme.of(context).colorScheme.surface,
            child: const TabBar(
              tabs: [
                Tab(text: 'Persona & voice'),
                Tab(text: 'Features'),
                Tab(text: 'Everyday'),
                Tab(text: 'Profiles'),
              ],
            ),
          ),
          Expanded(
            child: TabBarView(
              children: [
                PersonaVoiceScreen(client: client),
                SettingsBasicScreen(client: client),
                EverydayScreen(client: client),
                ProfilesScreen(client: client),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
