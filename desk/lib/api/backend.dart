/// The one seam to Vinkona's backend: a thin, typed client over the config
/// panel's localhost HTTP API (config_server.py).
///
/// THE RULE OF THIS APP: no logic lives here in Dart.  The backend owns every
/// decision (field audience levels, feature recipes, activity resolution); this
/// client fetches, displays, and posts back.  Anything smarter belongs in
/// config_server, where the web panel and the phone client can share it.
library;

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

/// One plain-language companion change inside a feature recipe.
class RecipeChange {
  final String path;
  final Object? value;
  final String label;
  RecipeChange({required this.path, required this.value, required this.label});

  factory RecipeChange.fromJson(Map<String, dynamic> j) => RecipeChange(
        path: j['path'] as String? ?? '',
        value: j['value'],
        label: j['label'] as String? ?? '',
      );
}

/// One direction (enable/disable) of a feature recipe: what flipping the
/// switch means, in plain words, and the companion settings that follow it.
class RecipeSide {
  final String summary;
  final String note;
  final List<RecipeChange> changes;
  RecipeSide({required this.summary, required this.note, required this.changes});

  factory RecipeSide.fromJson(Map<String, dynamic>? j) => RecipeSide(
        summary: j?['summary'] as String? ?? '',
        note: j?['note'] as String? ?? '',
        changes: ((j?['changes'] as List?) ?? const [])
            .whereType<Map<String, dynamic>>()
            .map(RecipeChange.fromJson)
            .toList(),
      );
}

/// A Basic-tier feature toggle with its preview-then-confirm story.
class FeatureRecipe {
  final String path; // the dotted config path of the on/off switch
  final String title;
  final RecipeSide enable;
  final RecipeSide disable;
  FeatureRecipe(
      {required this.path,
      required this.title,
      required this.enable,
      required this.disable});

  factory FeatureRecipe.fromJson(String path, Map<String, dynamic> j) =>
      FeatureRecipe(
        path: path,
        title: j['title'] as String? ?? path,
        enable: RecipeSide.fromJson(j['enable'] as Map<String, dynamic>?),
        disable: RecipeSide.fromJson(j['disable'] as Map<String, dynamic>?),
      );
}

/// What Vinkona is doing right now (the backend resolves the single headline).
class Activity {
  final String headline;
  final String doing;
  final bool interruptible;
  final bool sessionActive;
  Activity(
      {required this.headline,
      required this.doing,
      required this.interruptible,
      required this.sessionActive});

  factory Activity.fromJson(Map<String, dynamic> j) => Activity(
        headline: j['headline'] as String? ?? 'Unknown',
        doing: j['doing'] as String? ?? '',
        interruptible: j['interruptible'] as bool? ?? true,
        sessionActive:
            (j['session'] as Map<String, dynamic>?)?['active'] as bool? ?? false,
      );
}

/// One persona from the personas document.  A read-only view for pickers —
/// edits go through the WHOLE document ([BackendClient.savePersonas]), the
/// same contract the web panel uses.
class Persona {
  final String name; // the document key
  final String description;
  final String greeting;
  final String voice;
  Persona(
      {required this.name,
      required this.description,
      required this.greeting,
      required this.voice});
}

/// The personas document: who she can be, and who she is by default.
class PersonaDoc {
  final String defaultName;
  final List<Persona> personas;
  final Map<String, dynamic> raw; // the whole document, posted back on save
  PersonaDoc(
      {required this.defaultName, required this.personas, required this.raw});

  factory PersonaDoc.fromJson(Map<String, dynamic> j) {
    final ps = (j['personas'] as Map<String, dynamic>?) ?? const {};
    return PersonaDoc(
      defaultName: j['default'] as String? ?? '',
      personas: ps.entries
          .where((e) => e.value is Map<String, dynamic>)
          .map((e) {
        final p = e.value as Map<String, dynamic>;
        return Persona(
          name: e.key,
          description: p['description'] as String? ?? '',
          greeting: p['greeting'] as String? ?? '',
          voice: p['voice'] as String? ?? '',
        );
      }).toList(),
      raw: j,
    );
  }
}

/// One TTS engine from the backend's catalogue, with its install state and
/// preset voices (empty = the engine clones/designs voices instead).
class TtsEngine {
  final String key;
  final String label;
  final String footprint;
  final String note;
  final bool installed;
  final bool current;
  final List<String> voices;
  TtsEngine(
      {required this.key,
      required this.label,
      required this.footprint,
      required this.note,
      required this.installed,
      required this.current,
      required this.voices});

  factory TtsEngine.fromJson(Map<String, dynamic> j) => TtsEngine(
        key: j['key'] as String? ?? '',
        label: j['label'] as String? ?? '',
        footprint: j['footprint'] as String? ?? '',
        note: j['note'] as String? ?? '',
        installed: j['installed'] as bool? ?? false,
        current: j['current'] as bool? ?? false,
        voices: ((j['voices'] as List?) ?? const [])
            .map((v) => v.toString())
            .toList(),
      );
}

class TtsStatus {
  final String current;
  final String defaultVoice;
  final List<TtsEngine> engines;
  TtsStatus(
      {required this.current,
      required this.defaultVoice,
      required this.engines});

  TtsEngine? get currentEngine {
    for (final e in engines) {
      if (e.current) return e;
    }
    return null;
  }

  factory TtsStatus.fromJson(Map<String, dynamic> j) => TtsStatus(
        current: j['current'] as String? ?? '',
        defaultVoice: j['default_voice'] as String? ?? '',
        engines: ((j['engines'] as List?) ?? const [])
            .whereType<Map<String, dynamic>>()
            .map(TtsEngine.fromJson)
            .toList(),
      );
}

/// One memory+personas bundle ("profile") with its lightweight stats.
class ProfileInfo {
  final String name;
  final int memories; // -1 = unknown/unreadable
  final int personas;
  final int sizeBytes;
  final bool active;
  ProfileInfo(
      {required this.name,
      required this.memories,
      required this.personas,
      required this.sizeBytes,
      required this.active});

  factory ProfileInfo.fromJson(Map<String, dynamic> j) => ProfileInfo(
        name: j['name'] as String? ?? '',
        memories: (j['memories'] as num?)?.toInt() ?? -1,
        personas: (j['personas'] as num?)?.toInt() ?? 0,
        sizeBytes: (j['size'] as num?)?.toInt() ?? 0,
        active: j['active'] as bool? ?? false,
      );
}

class ProfilesStatus {
  final String active;
  final List<ProfileInfo> profiles;
  ProfilesStatus({required this.active, required this.profiles});

  factory ProfilesStatus.fromJson(Map<String, dynamic> j) => ProfilesStatus(
        active: j['active'] as String? ?? '',
        profiles: ((j['profiles'] as List?) ?? const [])
            .whereType<Map<String, dynamic>>()
            .map(ProfileInfo.fromJson)
            .toList(),
      );
}

/// The audience-tier map: dotted config path -> basic | advanced | expert.
class FieldLevels {
  final Map<String, String> levels;
  final String defaultLevel;
  final Map<String, String> labels; // optional friendly names per path
  final Map<String, List<Object?>> choices; // fixed-choice fields (pickers)
  FieldLevels(
      {required this.levels,
      required this.defaultLevel,
      required this.labels,
      required this.choices});

  String levelFor(String path) {
    // Exact match wins; then glob prefixes ("vad.*"); then the default.
    final exact = levels[path];
    if (exact != null) return exact;
    for (final e in levels.entries) {
      if (e.key.endsWith('.*') &&
          path.startsWith(e.key.substring(0, e.key.length - 1))) {
        return e.value;
      }
    }
    return defaultLevel;
  }

  factory FieldLevels.fromJson(Map<String, dynamic> j) => FieldLevels(
        levels: ((j['levels'] as Map<String, dynamic>?) ?? const {})
            .map((k, v) => MapEntry(k, v as String)),
        defaultLevel: j['default'] as String? ?? 'advanced',
        labels: ((j['labels'] as Map<String, dynamic>?) ?? const {})
            .map((k, v) => MapEntry(k, v as String)),
        choices: ((j['choices'] as Map<String, dynamic>?) ?? const {})
            .map((k, v) => MapEntry(k, List<Object?>.from(v as List))),
      );
}

class BackendException implements Exception {
  final String message;
  BackendException(this.message);
  @override
  String toString() => message;
}

/// The client.  Constructed once; screens call and render.
class BackendClient {
  final String baseUrl;
  final http.Client _http;
  BackendClient({this.baseUrl = 'http://127.0.0.1:8090', http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  Future<Map<String, dynamic>> _getJson(String path) async {
    final r = await _http
        .get(Uri.parse('$baseUrl$path'))
        .timeout(const Duration(seconds: 6));
    if (r.statusCode != 200) {
      throw BackendException('HTTP ${r.statusCode} from $path');
    }
    final body = jsonDecode(r.body);
    if (body is! Map<String, dynamic>) {
      throw BackendException('$path returned a non-object');
    }
    return body;
  }

  /// The merged config (defaults ⊕ the user's overrides): what every settings
  /// surface edits.  Saved back WHOLE via [saveConfig] — same contract as the
  /// web panel.
  Future<Map<String, dynamic>> config() => _getJson('/api/config');

  Future<void> saveConfig(Map<String, dynamic> full) async {
    final r = await _http
        .post(Uri.parse('$baseUrl/api/config'), body: jsonEncode(full))
        .timeout(const Duration(seconds: 10));
    if (r.statusCode != 200) {
      throw BackendException('config save failed: HTTP ${r.statusCode}');
    }
  }

  Future<Map<String, dynamic>> _postJson(String path, Object body) async {
    final r = await _http
        .post(Uri.parse('$baseUrl$path'), body: jsonEncode(body))
        .timeout(const Duration(seconds: 10));
    if (r.statusCode != 200) {
      String detail = '';
      try {
        detail = (jsonDecode(r.body) as Map<String, dynamic>)['error'] ?? '';
      } catch (_) {}
      throw BackendException(
          'HTTP ${r.statusCode} from $path${detail.isEmpty ? "" : ": $detail"}');
    }
    final body2 = jsonDecode(r.body);
    return body2 is Map<String, dynamic> ? body2 : <String, dynamic>{};
  }

  Future<FieldLevels> fieldLevels() async =>
      FieldLevels.fromJson(await _getJson('/api/field_levels'));

  /// The personas document (who she can be).  Edited whole, like config.
  Future<PersonaDoc> personas() async =>
      PersonaDoc.fromJson(await _getJson('/api/personas'));

  Future<void> savePersonas(Map<String, dynamic> doc) =>
      _postJson('/api/personas', doc);

  Future<TtsStatus> tts() async => TtsStatus.fromJson(await _getJson('/api/tts'));

  /// Select a TTS engine.  [when] is 'next_restart' (default) or 'now' (the
  /// backend then restarts everything to reconcile the service set).
  Future<Map<String, dynamic>> ttsSelect(String engine,
          {String when = 'next_restart'}) =>
      _postJson('/api/tts/select', {'engine': engine, 'when': when});

  /// The memory+personas bundles and which one is live.
  Future<ProfilesStatus> profiles() async =>
      ProfilesStatus.fromJson(await _getJson('/api/profiles'));

  Future<void> profileSwitch(String name) =>
      _postJson('/api/profiles/switch', {'name': name});

  Future<void> profileCreate(String name) =>
      _postJson('/api/profiles/create', {'name': name});

  Future<List<FeatureRecipe>> featureRecipes() async {
    final j = await _getJson('/api/feature_recipes');
    final recipes = (j['recipes'] as Map<String, dynamic>?) ?? const {};
    return recipes.entries
        .where((e) => e.value is Map<String, dynamic>)
        .map((e) =>
            FeatureRecipe.fromJson(e.key, e.value as Map<String, dynamic>))
        .toList();
  }

  Future<Activity> activity() async =>
      Activity.fromJson(await _getJson('/api/activity'));

  /// Help text per dotted config path (extracted from config.py's comments).
  Future<Map<String, String>> help() async {
    final j = await _getJson('/api/help');
    final h = (j['help'] as Map<String, dynamic>?) ?? const {};
    return h.map((k, v) => MapEntry(k, v.toString()));
  }

  /// True when the backend answers at all — the connection card's probe.
  Future<bool> reachable() async {
    try {
      await activity();
      return true;
    } catch (_) {
      return false;
    }
  }

  void close() => _http.close();
}

/// Read or write a dotted path inside the (mutable) merged-config map.
Object? configGet(Map<String, dynamic> cfg, String dotted) {
  Object? cur = cfg;
  for (final part in dotted.split('.')) {
    if (cur is Map<String, dynamic>) {
      cur = cur[part];
    } else {
      return null;
    }
  }
  return cur;
}

void configSet(Map<String, dynamic> cfg, String dotted, Object? value) {
  final parts = dotted.split('.');
  Map<String, dynamic> cur = cfg;
  for (final part in parts.sublist(0, parts.length - 1)) {
    final next = cur[part];
    if (next is Map<String, dynamic>) {
      cur = next;
    } else {
      final made = <String, dynamic>{};
      cur[part] = made;
      cur = made;
    }
  }
  cur[parts.last] = value;
}
