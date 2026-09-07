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
import { Ionicons } from '@expo/vector-icons';
import type { PackLeaderboard, PacksCatalog, SocialPack } from '../../api/social-adventure';
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

function PackCard({
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
    <View style={[styles.packCard, selected && styles.packCardSelected]}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`Open ${pack.name} Pack`}
        style={styles.packCopy}
        onPress={onSelect}
      >
        <Text style={styles.packName}>{pack.name}</Text>
        <Text style={styles.meta}>
          {pack.regionKey ?? 'coarse region'} · {pack.memberCount}{' '}
          {pack.memberCount === 1 ? 'member' : 'members'}
        </Text>
      </Pressable>

      {pack.role === 'OWNER' ? (
        <View style={styles.ownerBadge}>
          <Text style={styles.ownerText}>Owner</Text>
        </View>
      ) : (
        <Pressable
          accessibilityRole="button"
          disabled={busy}
          style={styles.secondaryAction}
          onPress={pack.joined ? onLeave : onJoin}
        >
          {busy ? (
            <ActivityIndicator size="small" color={colors.primary[700]} />
          ) : (
            <Text style={styles.secondaryActionText}>{pack.joined ? 'Leave' : 'Join'}</Text>
          )}
        </Pressable>
      )}
    </View>
  );
}

function PackStandings({
  leaderboard,
  loading,
}: {
  leaderboard: PackLeaderboard | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <View style={styles.inlineLoading} accessibilityRole="progressbar">
        <ActivityIndicator size="small" color={colors.primary[600]} />
      </View>
    );
  }

  if (leaderboard && !leaderboard.cohortReady) {
    return (
      <View style={styles.cohortCard}>
        <Ionicons name="shield-outline" size={22} color={colors.primary[700]} />
        <View style={styles.flexCopy}>
          <Text style={styles.cohortTitle}>Building a privacy-safe cohort</Text>
          <Text style={styles.bodyText}>
            {leaderboard.message ?? 'Standings remain hidden until the server cohort is ready.'}
          </Text>
          <Text style={styles.cohortCount}>
            {leaderboard.pack.memberCount}/{leaderboard.minimumCohort} members
          </Text>
        </View>
      </View>
    );
  }

  if (!leaderboard?.cohortReady) return null;

  return (
    <View style={styles.rows}>
      {leaderboard.entries.map((entry) => (
        <View key={entry.userId} style={styles.rankRow}>
          <View style={styles.rankBadge}>
            <Text style={styles.rankText}>{entry.rank}</Text>
          </View>
          <View style={styles.flexCopy}>
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
        <View style={styles.heroIcon}>
          <Ionicons name="map-outline" size={23} color={colors.primary[700]} />
        </View>
        <Text style={styles.eyebrow}>SOCIAL ADVENTURE</Text>
        <Text style={styles.title}>Local Packs without tracking you.</Text>
        <Text style={styles.bodyText}>
          Choose a coarse community, not a coordinate. Pack rank never uses your home, route
          endpoints, live GPS, health, mileage, or dog performance.
        </Text>
        <View style={styles.boundaryCard}>
          <Ionicons name="shield-checkmark-outline" size={20} color={colors.success.dark} />
          <Text style={styles.boundaryText}>
            Local standings stay hidden until the server says the cohort is large enough. The app
            never estimates or reconstructs a private local rank.
          </Text>
        </View>
      </View>

      <View style={styles.sectionHeader}>
        <View>
          <Text style={styles.eyebrow}>OPT-IN COMMUNITIES</Text>
          <Text style={styles.sectionTitle}>Find a Pack</Text>
        </View>
        <Ionicons name="people-outline" size={22} color={colors.primary[600]} />
      </View>

      {!props.catalog?.packs.length ? (
        <View style={styles.quietCard}>
          <Text style={styles.quietTitle}>No local Packs yet.</Text>
          <Text style={styles.bodyText}>
            A quiet map is valid. You can start a broad-area Pack below.
          </Text>
        </View>
      ) : (
        <View style={styles.rows}>
          {props.catalog.packs.map((pack) => (
            <PackCard
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

      {props.selectedPack && (
        <View style={styles.leagueSection}>
          <Text style={styles.eyebrow}>PACK LEAGUE</Text>
          <Text style={styles.sectionTitle}>Human-side standings</Text>
          <Text style={styles.bodyText}>
            Breadth in Human Skill and bounded Adventure variety count. Repetition volume, likes,
            missed days, health, and exercise intensity do not.
          </Text>
          <PackStandings leaderboard={props.leaderboard} loading={props.leaderboardLoading} />
        </View>
      )}

      <View style={styles.createCard}>
        <View style={styles.createHeading}>
          <Ionicons name="add-circle-outline" size={22} color={colors.primary[700]} />
          <View style={styles.flexCopy}>
            <Text style={styles.createTitle}>Start a coarse-locality Pack</Text>
            <Text style={styles.bodyText}>
              Use a broad place people recognize. Do not enter an address, apartment complex,
              school, route, or exact meetup point.
            </Text>
          </View>
        </View>

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
          autoCapitalize="none"
          autoCorrect={false}
          maxLength={64}
          placeholder="south-bay-ca"
          placeholderTextColor={colors.gray[400]}
          style={styles.input}
        />

        <Pressable
          accessibilityRole="button"
          disabled={props.creating}
          style={[styles.primaryAction, props.creating && styles.disabled]}
          onPress={props.onCreatePack}
        >
          {props.creating ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.primaryActionText}>Create Pack</Text>
          )}
        </Pressable>
      </View>

      {props.catalog && (
        <Text style={styles.locationContract}>
          Server location contract: {props.catalog.locationContract}. Minimum cohort:{' '}
          {props.catalog.localMinimumCohort}.
        </Text>
      )}

      {props.error && (
        <View style={styles.errorCard} accessibilityRole="alert">
          <Ionicons name="alert-circle-outline" size={18} color={colors.error.dark} />
          <Text style={styles.errorText}>{props.error}</Text>
        </View>
      )}

      <Pressable accessibilityRole="button" style={styles.backButton} onPress={props.onBack}>
        <Text style={styles.backButtonText}>Back to Community</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 50 },
  hero: { borderRadius: 24, padding: 20, backgroundColor: colors.primary[50], borderWidth: 1, borderColor: colors.primary[100] },
  heroIcon: { width: 46, height: 46, borderRadius: 15, alignItems: 'center', justifyContent: 'center', backgroundColor: '#ffffff', marginBottom: 18 },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  title: { marginTop: 5, color: colors.gray[900], fontSize: 27, lineHeight: 34, fontWeight: '800' },
  sectionTitle: { marginTop: 3, color: colors.gray[900], fontSize: 20, fontWeight: '800' },
  bodyText: { marginTop: 6, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  boundaryCard: { marginTop: 16, padding: 13, borderRadius: 15, flexDirection: 'row', gap: 9, backgroundColor: colors.success.light },
  boundaryText: { flex: 1, color: colors.success.dark, fontSize: 12, lineHeight: 18 },
  sectionHeader: { marginTop: 28, marginBottom: 11, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' },
  quietCard: { padding: 18, borderRadius: 17, backgroundColor: '#ffffff' },
  quietTitle: { color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  rows: { gap: 9 },
  packCard: { minHeight: 76, padding: 14, borderRadius: 17, borderWidth: 1, borderColor: colors.gray[200], backgroundColor: '#ffffff', flexDirection: 'row', alignItems: 'center' },
  packCardSelected: { borderColor: colors.primary[300], backgroundColor: colors.primary[50] },
  packCopy: { flex: 1, paddingRight: 10 },
  packName: { color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  meta: { marginTop: 3, color: colors.gray[600], fontSize: 11 },
  ownerBadge: { borderRadius: 999, paddingHorizontal: 10, paddingVertical: 6, backgroundColor: colors.primary[100] },
  ownerText: { color: colors.primary[800], fontSize: 11, fontWeight: '800' },
  secondaryAction: { minWidth: 64, minHeight: 38, paddingHorizontal: 12, borderRadius: 11, borderWidth: 1, borderColor: colors.primary[200], alignItems: 'center', justifyContent: 'center', backgroundColor: '#ffffff' },
  secondaryActionText: { color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  leagueSection: { marginTop: 30 },
  inlineLoading: { minHeight: 76, alignItems: 'center', justifyContent: 'center' },
  cohortCard: { marginTop: 12, padding: 16, borderRadius: 17, backgroundColor: colors.primary[50], flexDirection: 'row', gap: 11 },
  flexCopy: { flex: 1 },
  cohortTitle: { color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  cohortCount: { marginTop: 8, color: colors.primary[700], fontSize: 12, fontWeight: '800' },
  rankRow: { minHeight: 64, padding: 11, borderRadius: 15, borderWidth: 1, borderColor: colors.gray[200], backgroundColor: '#ffffff', flexDirection: 'row', alignItems: 'center' },
  rankBadge: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.primary[100] },
  rankText: { color: colors.primary[800], fontSize: 13, fontWeight: '900' },
  handle: { color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  score: { color: colors.primary[700], fontSize: 18, fontWeight: '900' },
  createCard: { marginTop: 30, padding: 18, borderRadius: 20, borderWidth: 1, borderColor: colors.gray[200], backgroundColor: '#ffffff' },
  createHeading: { flexDirection: 'row', gap: 10 },
  createTitle: { color: colors.gray[900], fontSize: 16, fontWeight: '800' },
  fieldLabel: { marginTop: 15, marginBottom: 6, color: colors.gray[800], fontSize: 12, fontWeight: '800' },
  input: { minHeight: 47, borderWidth: 1, borderColor: colors.gray[300], borderRadius: 12, paddingHorizontal: 12, color: colors.gray[900], backgroundColor: colors.gray[50] },
  primaryAction: { minHeight: 49, borderRadius: 13, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.primary[600], marginTop: 17 },
  primaryActionText: { color: '#ffffff', fontSize: 14, fontWeight: '800' },
  disabled: { opacity: 0.55 },
  locationContract: { marginTop: 13, color: colors.gray[500], fontSize: 11, lineHeight: 16, textAlign: 'center' },
  errorCard: { marginTop: 14, padding: 12, borderRadius: 12, flexDirection: 'row', gap: 8, backgroundColor: colors.error.light },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  backButton: { minHeight: 45, marginTop: 20, alignItems: 'center', justifyContent: 'center' },
  backButtonText: { color: colors.primary[700], fontSize: 13, fontWeight: '800' },
});