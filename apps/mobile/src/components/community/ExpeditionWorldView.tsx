import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type {
  ExpeditionJournal,
  ExpeditionObjective,
  ExpeditionProjection,
} from '../../api/expeditions';
import type { SocialPack } from '../../api/social-adventure';
import { colors } from '../../theme/tokens';
import { ExpeditionFieldJournalView } from './ExpeditionFieldJournalView';

type Props = {
  globalProjection: ExpeditionProjection | null;
  packProjection: ExpeditionProjection | null;
  journal: ExpeditionJournal | null;
  joinedPacks: SocialPack[];
  selectedScope: 'GLOBAL' | string;
  packLoading: boolean;
  refreshing: boolean;
  error: string | null;
  onRefresh: () => void;
  onSelectGlobal: () => void;
  onSelectPack: (packId: string) => void;
  onOpenSkillcraft: () => void;
  onOpenPacks: () => void;
};

type LandmarkSpec = {
  key: ExpeditionObjective['key'];
  place: string;
  icon: keyof typeof Ionicons.glyphMap;
  atmosphere: string;
};

const LANDMARKS: LandmarkSpec[] = [
  {
    key: 'SNIFF_EXPLORE',
    place: 'Wandering Grove',
    icon: 'leaf-outline',
    atmosphere: 'A place for varied exploration and enrichment, never a mileage race.',
  },
  {
    key: 'RECOVERY_COUNTS',
    place: 'Resting Hollow',
    icon: 'moon-outline',
    atmosphere: 'A quiet clearing that makes recovery visible as real participation.',
  },
  {
    key: 'READ_THE_ROOM',
    place: 'Signal Observatory',
    icon: 'bulb-outline',
    atmosphere: 'A lookout built from distinct Human Skill rooms, not practice-score magnitude.',
  },
];

function formatSeason(start: string, end: string) {
  const startsAt = new Date(start);
  const endsAt = new Date(end);
  if (!Number.isFinite(startsAt.getTime()) || !Number.isFinite(endsAt.getTime())) {
    return 'Server-defined weekly season';
  }
  const formatter = new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  });
  return `${formatter.format(startsAt)} – ${formatter.format(endsAt)}`;
}

function ScopeChip({
  label,
  selected,
  onPress,
}: {
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected }}
      onPress={onPress}
      style={[styles.scopeChip, selected && styles.scopeChipSelected]}
    >
      <Text style={[styles.scopeChipText, selected && styles.scopeChipTextSelected]}>{label}</Text>
    </Pressable>
  );
}

function Landmark({
  objective,
  spec,
}: {
  objective: ExpeditionObjective | null;
  spec: LandmarkSpec;
}) {
  const participated = Boolean(objective && objective.myContribution > 0);

  return (
    <View style={styles.landmarkCard}>
      <View style={[styles.landmarkIcon, participated && styles.landmarkIconMine]}>
        <Ionicons
          name={spec.icon}
          size={27}
          color={participated ? colors.primary[700] : colors.gray[600]}
        />
      </View>
      <View style={styles.landmarkCopy}>
        <View style={styles.landmarkHeading}>
          <View style={styles.flex}>
            <Text style={styles.placeName}>{spec.place}</Text>
            <Text style={styles.objectiveTitle}>
              {objective?.title ?? 'Server objective unavailable'}
            </Text>
          </View>
          {participated && (
            <View style={styles.myMark}>
              <Ionicons name="paw-outline" size={13} color={colors.primary[700]} />
              <Text style={styles.myMarkText}>Your mark</Text>
            </View>
          )}
        </View>

        <Text style={styles.atmosphere}>{spec.atmosphere}</Text>

        {objective ? (
          <>
            <View style={styles.countRow}>
              <View style={styles.countCell}>
                <Text style={styles.countValue}>{objective.total}</Text>
                <Text style={styles.countLabel}>shared marks</Text>
              </View>
              <View style={styles.countCell}>
                <Text style={styles.countValue}>{objective.contributors}</Text>
                <Text style={styles.countLabel}>contributors</Text>
              </View>
              <View style={styles.countCell}>
                <Text style={styles.countValue}>{objective.myContribution}</Text>
                <Text style={styles.countLabel}>yours</Text>
              </View>
            </View>
            <View style={styles.calibrationNote}>
              <Ionicons name="flask-outline" size={15} color={colors.primary[700]} />
              <Text style={styles.calibrationText}>
                {objective.status === 'CALIBRATING'
                  ? 'Calibrating, no completion target yet.'
                  : 'Server authority has not supplied a supported target state.'}
              </Text>
            </View>
          </>
        ) : (
          <Text style={styles.unavailableText}>
            Woof will not estimate this landmark from another score or local activity history.
          </Text>
        )}
      </View>
    </View>
  );
}

export function ExpeditionWorldView(props: Props) {
  const projection =
    props.selectedScope === 'GLOBAL' ? props.globalProjection : props.packProjection;

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={props.refreshing} onRefresh={props.onRefresh} />}
    >
      <View style={styles.hero}>
        <Text style={styles.eyebrow}>WEEKLY EXPEDITION</Text>
        <Text style={styles.heroTitle}>Build a shared world, not a bigger score.</Text>
        <Text style={styles.heroBody}>
          Useful variety leaves marks across the landscape. The scene is playful; what counts stays
          bounded and server-authored.
        </Text>

        <View style={styles.worldScene} accessible accessibilityLabel="Shared Expedition landscape">
          <View style={styles.skyOrb} />
          <View style={styles.hillBack} />
          <View style={styles.hillFront} />
          <View style={styles.sceneLandmarks}>
            <View style={styles.sceneNode}>
              <Ionicons name="leaf-outline" size={25} color={colors.success.dark} />
              <Text style={styles.sceneLabel}>Grove</Text>
            </View>
            <View style={styles.sceneNode}>
              <Ionicons name="moon-outline" size={25} color={colors.primary[700]} />
              <Text style={styles.sceneLabel}>Hollow</Text>
            </View>
            <View style={styles.sceneNode}>
              <Ionicons name="bulb-outline" size={25} color={colors.warning.dark} />
              <Text style={styles.sceneLabel}>Observatory</Text>
            </View>
          </View>
        </View>

        {projection && (
          <Text style={styles.seasonCopy}>
            {formatSeason(projection.season.startsAt, projection.season.endsAt)} ·{' '}
            {projection.scope === 'GLOBAL'
              ? 'Everyone together'
              : (projection.pack?.name ?? 'Pack')}
          </Text>
        )}
      </View>

      <View style={styles.scopeSection}>
        <Text style={styles.sectionLabel}>CHOOSE A VIEW</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.scopeRow}
        >
          <ScopeChip
            label="Everyone"
            selected={props.selectedScope === 'GLOBAL'}
            onPress={props.onSelectGlobal}
          />
          {props.joinedPacks.map((pack) => (
            <ScopeChip
              key={pack.id}
              label={pack.name}
              selected={props.selectedScope === pack.id}
              onPress={() => props.onSelectPack(pack.id)}
            />
          ))}
        </ScrollView>
        {props.joinedPacks.length === 0 && (
          <Text style={styles.scopeHint}>
            No joined Pack is shown. Woof does not infer local membership from your location.
          </Text>
        )}
      </View>

      {props.error && (
        <View style={styles.errorCard} accessibilityRole="alert">
          <Ionicons name="cloud-offline-outline" size={19} color={colors.error.dark} />
          <Text style={styles.errorText}>{props.error}</Text>
        </View>
      )}

      {props.selectedScope !== 'GLOBAL' && props.packLoading ? (
        <View style={styles.loadingCard} accessibilityRole="progressbar">
          <ActivityIndicator color={colors.primary[600]} />
          <Text style={styles.loadingText}>Reading this Pack’s server-issued world…</Text>
        </View>
      ) : projection ? (
        <View style={styles.landmarksSection}>
          <Text style={styles.sectionLabel}>LANDMARKS</Text>
          <Text style={styles.sectionTitle}>A landscape with three kinds of contribution</Text>
          <Text style={styles.sectionBody}>
            Landmarks always exist. Counts show server-issued evidence, not unlock levels.
          </Text>
          {LANDMARKS.map((spec) => (
            <Landmark
              key={spec.key}
              spec={spec}
              objective={
                projection.objectives.find((objective) => objective.key === spec.key) ?? null
              }
            />
          ))}
        </View>
      ) : (
        <View style={styles.loadingCard}>
          <Ionicons name="shield-outline" size={24} color={colors.gray[600]} />
          <Text style={styles.unavailableTitle}>This Expedition view is unavailable.</Text>
          <Text style={styles.unavailableText}>
            Woof will not reconstruct shared progress from cached activity, league score, or local
            guesses.
          </Text>
        </View>
      )}

      <ExpeditionFieldJournalView journal={props.journal} />

      <View style={styles.boundaryCard}>
        <View style={styles.boundaryIcon}>
          <Ionicons name="shield-checkmark-outline" size={21} color={colors.primary[700]} />
        </View>
        <View style={styles.flex}>
          <Text style={styles.boundaryTitle}>The world is the game. Your dog is not.</Text>
          <Text style={styles.boundaryBody}>
            CARE, health state, distance, duration, intensity, missed days, likes, rankings, and
            repeated grinding add no Expedition progress. Recovery can count because listening can
            be the useful choice.
          </Text>
          <Text style={styles.noMeter}>
            No completion bar. This shared scene is not a checklist.
          </Text>
        </View>
      </View>

      <View style={styles.actionRow}>
        <Pressable
          accessibilityRole="button"
          onPress={props.onOpenSkillcraft}
          style={styles.primaryAction}
        >
          <Ionicons name="game-controller-outline" size={18} color="#ffffff" />
          <Text style={styles.primaryActionText}>Practice Skillcraft</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          onPress={props.onOpenPacks}
          style={styles.secondaryAction}
        >
          <Ionicons name="people-outline" size={18} color={colors.primary[700]} />
          <Text style={styles.secondaryActionText}>Explore Packs</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 16, paddingBottom: 40 },
  hero: {
    overflow: 'hidden',
    padding: 18,
    borderRadius: 24,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.3 },
  heroTitle: {
    marginTop: 5,
    color: colors.gray[900],
    fontSize: 27,
    lineHeight: 33,
    fontWeight: '900',
  },
  heroBody: { marginTop: 8, color: colors.gray[600], fontSize: 13, lineHeight: 20 },
  worldScene: {
    height: 168,
    marginTop: 17,
    overflow: 'hidden',
    borderRadius: 20,
    backgroundColor: '#eef8ff',
    borderWidth: 1,
    borderColor: '#dbeafe',
  },
  skyOrb: {
    position: 'absolute',
    right: 23,
    top: 18,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#fef3c7',
  },
  hillBack: {
    position: 'absolute',
    left: -35,
    right: 35,
    bottom: -38,
    height: 130,
    borderRadius: 90,
    backgroundColor: '#dcfce7',
    transform: [{ rotate: '-4deg' }],
  },
  hillFront: {
    position: 'absolute',
    left: 45,
    right: -40,
    bottom: -54,
    height: 128,
    borderRadius: 90,
    backgroundColor: '#ede9fe',
    transform: [{ rotate: '5deg' }],
  },
  sceneLandmarks: {
    position: 'absolute',
    left: 18,
    right: 18,
    bottom: 23,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
  },
  sceneNode: {
    minWidth: 72,
    paddingHorizontal: 9,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.9)',
  },
  sceneLabel: { marginTop: 4, color: colors.gray[700], fontSize: 10, fontWeight: '800' },
  seasonCopy: { marginTop: 11, color: colors.primary[800], fontSize: 11, fontWeight: '700' },
  scopeSection: { marginTop: 20 },
  sectionLabel: { color: colors.gray[500], fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  scopeRow: { gap: 8, paddingTop: 9, paddingRight: 18 },
  scopeChip: {
    paddingHorizontal: 13,
    paddingVertical: 9,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: '#ffffff',
  },
  scopeChipSelected: { borderColor: colors.primary[300], backgroundColor: colors.primary[100] },
  scopeChipText: { color: colors.gray[700], fontSize: 12, fontWeight: '700' },
  scopeChipTextSelected: { color: colors.primary[800] },
  scopeHint: { marginTop: 8, color: colors.gray[500], fontSize: 11, lineHeight: 17 },
  errorCard: {
    marginTop: 14,
    padding: 13,
    flexDirection: 'row',
    gap: 9,
    borderRadius: 14,
    backgroundColor: colors.error.light,
  },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 11, lineHeight: 17 },
  landmarksSection: { marginTop: 24 },
  sectionTitle: { marginTop: 4, color: colors.gray[900], fontSize: 20, fontWeight: '900' },
  sectionBody: { marginTop: 5, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  landmarkCard: {
    marginTop: 11,
    padding: 15,
    flexDirection: 'row',
    gap: 12,
    borderRadius: 19,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  landmarkIcon: {
    width: 50,
    height: 50,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 17,
    backgroundColor: colors.gray[100],
  },
  landmarkIconMine: { backgroundColor: colors.primary[100] },
  landmarkCopy: { flex: 1 },
  landmarkHeading: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  flex: { flex: 1 },
  placeName: { color: colors.gray[900], fontSize: 16, fontWeight: '900' },
  objectiveTitle: { marginTop: 2, color: colors.primary[700], fontSize: 10, fontWeight: '800' },
  myMark: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 7,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: colors.primary[50],
  },
  myMarkText: { color: colors.primary[800], fontSize: 9, fontWeight: '800' },
  atmosphere: { marginTop: 7, color: colors.gray[600], fontSize: 11, lineHeight: 17 },
  countRow: { marginTop: 11, flexDirection: 'row', gap: 7 },
  countCell: { flex: 1, padding: 8, borderRadius: 11, backgroundColor: colors.gray[50] },
  countValue: { color: colors.gray[900], fontSize: 17, fontWeight: '900' },
  countLabel: { marginTop: 1, color: colors.gray[500], fontSize: 8 },
  calibrationNote: {
    marginTop: 9,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    padding: 8,
    borderRadius: 10,
    backgroundColor: colors.primary[50],
  },
  calibrationText: {
    flex: 1,
    color: colors.primary[800],
    fontSize: 10,
    lineHeight: 15,
    fontWeight: '700',
  },
  loadingCard: {
    marginTop: 22,
    padding: 24,
    alignItems: 'center',
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  loadingText: { marginTop: 8, color: colors.gray[600], fontSize: 12 },
  unavailableTitle: { marginTop: 8, color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  unavailableText: {
    marginTop: 5,
    color: colors.gray[600],
    fontSize: 11,
    lineHeight: 17,
    textAlign: 'center',
  },
  boundaryCard: {
    marginTop: 22,
    padding: 15,
    flexDirection: 'row',
    gap: 11,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  boundaryIcon: {
    width: 38,
    height: 38,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 13,
    backgroundColor: colors.primary[50],
  },
  boundaryTitle: { color: colors.gray[900], fontSize: 14, fontWeight: '900' },
  boundaryBody: { marginTop: 5, color: colors.gray[600], fontSize: 11, lineHeight: 17 },
  noMeter: { marginTop: 7, color: colors.primary[800], fontSize: 10, fontWeight: '800' },
  actionRow: { marginTop: 16, flexDirection: 'row', gap: 9 },
  primaryAction: {
    flex: 1,
    minHeight: 46,
    flexDirection: 'row',
    gap: 7,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 14,
    backgroundColor: colors.primary[600],
  },
  primaryActionText: { color: '#ffffff', fontSize: 11, fontWeight: '800' },
  secondaryAction: {
    flex: 1,
    minHeight: 46,
    flexDirection: 'row',
    gap: 7,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.primary[200],
    backgroundColor: '#ffffff',
  },
  secondaryActionText: { color: colors.primary[800], fontSize: 11, fontWeight: '800' },
});
