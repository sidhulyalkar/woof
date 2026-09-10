import React from 'react';
import { Ionicons } from '@expo/vector-icons';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import type {
  ExpeditionJournal,
  ExpeditionJournalEntry,
  ExpeditionObjectiveKey,
} from '../../api/expeditions';
import { colors } from '../../theme/tokens';

type Props = {
  journal: ExpeditionJournal | null;
  error?: string | null;
};

type JournalLandmark = {
  place: string;
  icon: keyof typeof Ionicons.glyphMap;
};

const JOURNAL_LANDMARKS: Record<ExpeditionObjectiveKey, JournalLandmark> = {
  SNIFF_EXPLORE: { place: 'Wandering Grove', icon: 'leaf-outline' },
  RECOVERY_COUNTS: { place: 'Resting Hollow', icon: 'moon-outline' },
  READ_THE_ROOM: { place: 'Signal Observatory', icon: 'bulb-outline' },
};

function formatWeek(startsAt: string) {
  const date = new Date(startsAt);
  if (!Number.isFinite(date.getTime())) return 'Recorded week';
  const formatter = new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  });
  return `Week of ${formatter.format(date)}`;
}

function FieldNote({ entry }: { entry: ExpeditionJournalEntry }) {
  return (
    <View style={styles.noteCard}>
      <View style={styles.noteHeader}>
        <View style={styles.postmark}>
          <Ionicons name="paw-outline" size={15} color={colors.primary[700]} />
        </View>
        <View style={styles.noteHeading}>
          <Text style={styles.noteState}>{entry.state === 'ACTIVE' ? 'THIS WEEK' : 'FIELD NOTE'}</Text>
          <Text style={styles.noteDate}>{formatWeek(entry.season.startsAt)}</Text>
        </View>
      </View>

      <View style={styles.stamps}>
        {entry.landmarks.map((landmark) => {
          const spec = JOURNAL_LANDMARKS[landmark.key];
          return (
            <View key={landmark.key} style={styles.stamp}>
              <Ionicons name={spec.icon} size={18} color={colors.primary[700]} />
              <View style={styles.stampCopy}>
                <Text style={styles.stampPlace}>{spec.place}</Text>
                <Text style={styles.stampTitle}>{landmark.title}</Text>
              </View>
            </View>
          );
        })}
      </View>

      <Text style={styles.noteFoot}>
        {entry.state === 'ACTIVE'
          ? 'This page reflects verified moments so far. There is nothing you need to fill.'
          : 'This page records what happened. Blank space is part of the memory.'}
      </Text>
    </View>
  );
}

export function ExpeditionFieldJournalView({ journal, error }: Props) {
  return (
    <View style={styles.section}>
      <Text style={styles.eyebrow}>FIELD JOURNAL</Text>
      <Text style={styles.title}>Pages from worlds you helped inhabit</Text>
      <Text style={styles.body}>
        A field note remembers which kinds of Expedition moments actually occurred. Repeating a
        moment does not make a bigger stamp, and a page is never graded for being full.
      </Text>

      {error && journal && (
        <View style={styles.noticeCard} accessibilityRole="alert">
          <Ionicons name="cloud-offline-outline" size={18} color={colors.gray[600]} />
          <Text style={styles.noticeText}>{error}</Text>
        </View>
      )}

      {!journal ? (
        <View style={styles.quietCard} accessibilityRole={error ? 'alert' : undefined}>
          <Ionicons name="book-outline" size={23} color={colors.gray[500]} />
          <Text style={styles.quietTitle}>{error ? 'Field notes could not refresh.' : 'No field notes yet.'}</Text>
          <Text style={styles.quietBody}>
            {error
              ? 'Your shared world is still available. Woof will leave history blank rather than guess.'
              : 'Nothing is overdue and there is nothing to catch up on. A page appears only from verified Expedition moments.'}
          </Text>
        </View>
      ) : journal.entries.length === 0 ? (
        <View style={styles.quietCard}>
          <Ionicons name="leaf-outline" size={23} color={colors.primary[600]} />
          <Text style={styles.quietTitle}>No field notes yet.</Text>
          <Text style={styles.quietBody}>
            Nothing is overdue and there is nothing to catch up on. A page appears only from verified
            Expedition moments.
          </Text>
        </View>
      ) : (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.notesRow}
        >
          {journal.entries.map((entry) => (
            <FieldNote key={entry.season.key} entry={entry} />
          ))}
        </ScrollView>
      )}

      {journal && (
        <Text style={styles.coverage}>
          Recent journal coverage: up to {journal.coverage.maxSeasons} participated weeks. Missing
          older pages are not presented as non-participation.
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  section: { marginTop: 26 },
  eyebrow: { color: colors.gray[500], fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  title: { marginTop: 4, color: colors.gray[900], fontSize: 20, fontWeight: '900' },
  body: { marginTop: 6, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  noticeCard: {
    marginTop: 12,
    padding: 12,
    borderRadius: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: colors.gray[50],
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  noticeText: { flex: 1, color: colors.gray[600], fontSize: 11, lineHeight: 17 },
  notesRow: { gap: 11, paddingTop: 12, paddingRight: 18 },
  noteCard: {
    width: 286,
    minHeight: 214,
    padding: 16,
    borderRadius: 20,
    backgroundColor: '#fffdf7',
    borderWidth: 1,
    borderColor: '#eee7d7',
  },
  noteHeader: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  postmark: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.primary[200],
    backgroundColor: colors.primary[50],
  },
  noteHeading: { flex: 1 },
  noteState: { color: colors.primary[700], fontSize: 9, fontWeight: '900', letterSpacing: 1.1 },
  noteDate: { marginTop: 2, color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  stamps: { marginTop: 15, gap: 8 },
  stamp: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    padding: 10,
    borderRadius: 13,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  stampCopy: { flex: 1 },
  stampPlace: { color: colors.gray[900], fontSize: 12, fontWeight: '800' },
  stampTitle: { marginTop: 1, color: colors.gray[500], fontSize: 10 },
  noteFoot: { marginTop: 14, color: colors.gray[500], fontSize: 10, lineHeight: 15 },
  quietCard: {
    marginTop: 12,
    padding: 18,
    borderRadius: 18,
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  quietTitle: { marginTop: 8, color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  quietBody: {
    marginTop: 5,
    color: colors.gray[600],
    fontSize: 11,
    lineHeight: 17,
    textAlign: 'center',
  },
  coverage: { marginTop: 9, color: colors.gray[500], fontSize: 10, lineHeight: 15 },
});
