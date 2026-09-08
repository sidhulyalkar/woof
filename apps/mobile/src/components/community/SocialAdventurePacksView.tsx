import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import type {
  PackLeaderboard,
  PacksCatalog,
  SocialPack,
} from '../../api/social-adventure';
import { colors } from '../../theme/tokens';

type Props = {
  catalog: PacksCatalog | null;
  selectedPack: SocialPack | null;
  leaderboard: PackLeaderboard | null;
  leaderboardLoading: boolean;
  actionId: string | null;
  creating: boolean;
  name: string;
  regionKey: string;
  error: string | null;
  onSelectPack: (packId: string) => void;
  onJoinPack: (pack: SocialPack) => void;
  onLeavePack: (pack: SocialPack) => void;
  onNameChange: (value: string) => void;
  onRegionChange: (value: string) => void;
  onCreatePack: () => void;
  onBack: () => void;
};

function PackRow({
  pack,
  selected,
  busy,
  onSelect,
  onJoin,
  onLeave,
}: {
  pack: SocialPack;
  selected: boolean;
  busy: boolean;
  onSelect: () => void;
  onJoin: () => void;
  onLeave: () => void;
}) {
  return (
    <View style={[styles.packRow, selected && styles.packRowSelected]}>
      <Pressable accessibilityRole="button" onPress={onSelect} style={styles.packOpen}>
        <Text style={styles.packName}>{pack.name}</Text>
        <Text style={styles.meta}>
          {pack.regionKey ?? 'coarse region'} · {pack.memberCount}{' '}
          {pack.memberCount === 1 ? 'member' : 'members'}
        </Text>
      </Pressable>

      {pack.role === 'OWNER' ? (
        <View style={styles.badge}>
          <Text style={styles.badgeText}>Owner</Text>
        </View>
      ) : (
        <Pressable
          accessibilityRole="button"
          disabled={busy}
          onPress={pack.joined ? onLeave : onJoin}
          style={[styles.secondaryButton, busy && styles.disabled]}
        >
          {busy ? (
            <ActivityIndicator size="small" color={colors.primary[700]} />
          ) : (
            <Text style={styles.secondaryButtonText}>{pack.joined ? 'Leave' : 'Join'}</Text>
          )}
        </Pressable>
      )}
    </View>
  );
}

function Standings({
  selectedPack,
  leaderboard,
  loading,
}: {
  selectedPack: SocialPack;
  leaderboard: PackLeaderboard | null;
  loading: boolean;
}) {
  return (
    <View style={styles.card}>
      <Text style={styles.eyebrow}>PACK LEAGUE</Text>
      <Text style={styles.sectionTitle}>{selectedPack.name}</Text>
      <Text style={styles.body}>
        Breadth in Human Skill and bounded Adventure variety count. Repetition, likes, missed
        days, health, and exercise intensity do not.
      </Text>

      {loading ? (
        <ActivityIndicator style={styles.inlineLoading} color={colors.primary[600]} />
      ) : leaderboard && !leaderboard.cohortReady ? (
        <View style={styles.quietBox}>
          <Text style={styles.quietTitle}>Building a privacy-safe cohort</Text>
          <Text style={styles.smallCopy}>
            {leaderboard.message ?? 'Standings remain hidden until the server cohort is ready.'}
          </Text>
          <Text style={styles.cohortCount}>
            {leaderboard.pack.memberCount}/{leaderboard.minimumCohort} members
          </Text>
        </View>
      ) : leaderboard?.cohortReady ? (
        <View style={styles.rankList}>
          {leaderboard.entries.map((entry) => (
            <View key={entry.userId} style={styles.rankRow}>
              <Text style={styles.rankNumber}>#{entry.rank}</Text>
              <View style={styles.flex}>
                <Text style={styles.handle}>@{entry.handle}</Text>
                <Text style={styles.meta}>
                  {entry.components.humanSkill.score} skill ·{' '}
                  {entry.components.adventureVariety.pathways.length} pathways
                </Text>
              </View>
              <Text style={styles.score}>{entry.score}</Text>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}

export function SocialAdventurePacksView(props: Props) {
  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      <View style={styles.hero}>
        <Pressable accessibilityRole="button" onPress={props.onBack} style={styles.backButton}>
          <Text style={styles.backButtonText}>Back</Text>
        </Pressable>
        <Text style={styles.eyebrow}>SOCIAL ADVENTURE</Text>
        <Text style={styles.heroTitle}>Local Packs without tracking you.</Text>
        <Text style={styles.body}>
          Choose a coarse community, not a coordinate. Pack rank never uses your home, route
          endpoints, live GPS, health, mileage, or dog performance.
        </Text>
        <View style={styles.quietBox}>
          <Text style={styles.quietTitle}>Privacy floor</Text>
          <Text style={styles.smallCopy}>
            The app never estimates or reconstructs a private local rank. Standings appear only
            when the server says the cohort is large enough.
          </Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.eyebrow}>OPT-IN COMMUNITIES</Text>
        <Text style={styles.sectionTitle}>Find a Pack</Text>
        {props.catalog && (
          <Text style={styles.disclaimer}>{props.catalog.locationContract}</Text>
        )}

        {!props.catalog?.packs.length ? (
          <View style={styles.quietBox}>
            <Text style={styles.quietTitle}>No local Packs yet.</Text>
            <Text style={styles.smallCopy}>
              A quiet map is valid. You can start a broad-area Pack below.
            </Text>
          </View>
        ) : (
          <View style={styles.packList}>
            {props.catalog.packs.map((pack) => (
              <PackRow
                key={pack.id}
                pack={pack}
                selected={pack.id === props.selectedPack?.id}
                busy={props.actionId === pack.id}
                onSelect={() => props.onSelectPack(pack.id)}
                onJoin={() => props.onJoinPack(pack)}
                onLeave={() => props.onLeavePack(pack)}
              />
            ))}
          </View>
        )}
      </View>

      {props.selectedPack && (
        <Standings
          selectedPack={props.selectedPack}
          leaderboard={props.leaderboard}
          loading={props.leaderboardLoading}
        />
      )}

      <View style={styles.card}>
        <Text style={styles.eyebrow}>CREATE A PACK</Text>
        <Text style={styles.sectionTitle}>Start a coarse-locality Pack</Text>
        <Text style={styles.body}>
          Use a broad place people recognize. Do not enter an address, apartment complex, school,
          route, or exact meetup point.
        </Text>

        <Text style={styles.fieldLabel}>Pack name</Text>
        <TextInput
          value={props.name}
          onChangeText={props.onNameChange}
          maxLength={64}
          placeholder="South Bay Adventure Pack"
          placeholderTextColor={colors.gray[400]}
          style={styles.input}
        />

        <Text style={styles.fieldLabel}>Coarse region</Text>
        <TextInput
          value={props.regionKey}
          onChangeText={props.onRegionChange}
          maxLength={80}
          autoCapitalize="none"
          placeholder="south-bay-ca"
          placeholderTextColor={colors.gray[400]}
          style={styles.input}
        />

        <Pressable
          accessibilityRole="button"
          disabled={props.creating}
          onPress={props.onCreatePack}
          style={[styles.primaryButton, props.creating && styles.disabled]}
        >
          {props.creating ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.primaryButtonText}>Create Pack</Text>
          )}
        </Pressable>
      </View>

      {props.error && (
        <View style={styles.errorBox} accessibilityRole="alert">
          <Text style={styles.errorText}>{props.error}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 14, paddingBottom: 80 },
  hero: {
    padding: 18,
    borderRadius: 22,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  card: {
    marginTop: 14,
    padding: 16,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  backButton: { alignSelf: 'flex-start', marginBottom: 10, paddingVertical: 4 },
  backButtonText: { color: colors.primary[700], fontSize: 12, fontWeight: '800' },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  heroTitle: { marginTop: 5, color: colors.gray[900], fontSize: 27, lineHeight: 33, fontWeight: '800' },
  sectionTitle: { marginTop: 3, color: colors.gray[900], fontSize: 20, fontWeight: '800' },
  body: { marginTop: 7, color: colors.gray[600], fontSize: 13, lineHeight: 20 },
  smallCopy: { marginTop: 4, color: colors.gray[600], fontSize: 11, lineHeight: 17 },
  disclaimer: { marginTop: 7, color: colors.gray[500], fontSize: 10 },
  quietBox: { marginTop: 12, padding: 12, borderRadius: 12, backgroundColor: colors.gray[50] },
  quietTitle: { color: colors.gray[900], fontSize: 13, fontWeight: '800' },
  packList: { marginTop: 12, gap: 8 },
  packRow: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 10, borderRadius: 12, backgroundColor: colors.gray[50], borderWidth: 1, borderColor: colors.gray[100] },
  packRowSelected: { borderColor: colors.primary[300], backgroundColor: colors.primary[50] },
  packOpen: { flex: 1, paddingVertical: 3 },
  packName: { color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  meta: { marginTop: 2, color: colors.gray[500], fontSize: 10 },
  badge: { paddingHorizontal: 9, paddingVertical: 6, borderRadius: 999, backgroundColor: colors.primary[100] },
  badgeText: { color: colors.primary[800], fontSize: 10, fontWeight: '800' },
  secondaryButton: { paddingHorizontal: 11, paddingVertical: 7, borderRadius: 999, borderWidth: 1, borderColor: colors.primary[200], backgroundColor: '#ffffff' },
  secondaryButtonText: { color: colors.primary[800], fontSize: 10, fontWeight: '800' },
  inlineLoading: { marginTop: 14 },
  cohortCount: { marginTop: 7, color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  rankList: { marginTop: 12, gap: 7 },
  rankRow: { flexDirection: 'row', alignItems: 'center', gap: 9, padding: 9, borderRadius: 11, backgroundColor: colors.gray[50] },
  rankNumber: { width: 32, color: colors.primary[800], fontWeight: '900' },
  flex: { flex: 1 },
  handle: { color: colors.gray[900], fontSize: 13, fontWeight: '800' },
  score: { color: colors.primary[700], fontSize: 16, fontWeight: '900' },
  fieldLabel: { marginTop: 13, marginBottom: 5, color: colors.gray[700], fontSize: 11, fontWeight: '800' },
  input: { minHeight: 46, paddingHorizontal: 12, borderRadius: 11, borderWidth: 1, borderColor: colors.gray[200], backgroundColor: '#ffffff', color: colors.gray[900] },
  primaryButton: { marginTop: 14, padding: 13, borderRadius: 12, alignItems: 'center', backgroundColor: colors.primary[600] },
  primaryButtonText: { color: '#ffffff', fontWeight: '800' },
  disabled: { opacity: 0.55 },
  errorBox: { marginTop: 14, padding: 12, borderRadius: 12, backgroundColor: colors.error.light },
  errorText: { color: colors.error.dark, fontSize: 12, lineHeight: 17 },
});