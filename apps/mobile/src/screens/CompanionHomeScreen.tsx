import React, { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { StackScreenProps } from '@react-navigation/stack';
import { companionApi, type CompanionState } from '../api/companion';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'CompanionHome'> & {
  onResolved?: (state: CompanionState) => void;
};

const links: {
  route: 'Expedition' | 'Skillcraft' | 'Packs' | 'CommunityStandalone' | 'Events' | 'Profile';
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
}[] = [
  {
    route: 'Expedition',
    title: 'Shared Expedition',
    description:
      'Step into the cooperative world. Human-side participation does not require a dog.',
    icon: 'map-outline',
  },
  {
    route: 'Skillcraft',
    title: 'Skillcraft',
    description: 'Practice observation, setup, pairing, and timing before a real dog is involved.',
    icon: 'game-controller-outline',
  },
  {
    route: 'Packs',
    title: 'Packs',
    description:
      'Find small communities to learn and explore with, without borrowing pet authority.',
    icon: 'people-circle-outline',
  },
  {
    route: 'CommunityStandalone',
    title: 'Community',
    description: 'See shared moments, local plans, and people across Woof.',
    icon: 'people-outline',
  },
  {
    route: 'Events',
    title: 'Events',
    description: 'Explore community plans without claiming access to a pet.',
    icon: 'calendar-outline',
  },
  {
    route: 'Profile',
    title: 'You',
    description: 'Account, privacy, and deletion controls stay available.',
    icon: 'person-circle-outline',
  },
];

export default function CompanionHomeScreen({ navigation, onResolved }: Props) {
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const becomeGuardian = async () => {
    setSwitching(true);
    setError(null);
    try {
      const state = await companionApi.updateMode('PET_GUARDIAN');
      onResolved?.(state);
    } catch {
      setError('Woof could not change your starting role. No pet relationship was created.');
    } finally {
      setSwitching(false);
    }
  };

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.heroIcon}>
        <Ionicons name="heart-outline" size={25} color={colors.primary[700]} />
      </View>
      <Text style={styles.eyebrow}>COMPANION MODE</Text>
      <Text style={styles.title}>You can belong here before you have a dog.</Text>
      <Text style={styles.intro}>
        Explore the shared world, practice useful human skills, and meet people. Dog-specific Today,
        Compass, and Story open only when Woof can verify a real relationship you are allowed to
        see.
      </Text>

      <View style={styles.truthCard}>
        <Ionicons name="shield-checkmark-outline" size={21} color={colors.success.dark} />
        <View style={styles.truthCopy}>
          <Text style={styles.truthTitle}>Dog-specific spaces stay private</Text>
          <Text style={styles.truthText}>
            Companion mode gives you human-side places to learn and participate. It does not invent
            a pet relationship or unlock pet-specific Today, Compass, or Story.
          </Text>
        </View>
      </View>

      <Text style={styles.sectionTitle}>Start anywhere</Text>
      <Text style={styles.sectionIntro}>
        These paths are useful with or without a dog of your own.
      </Text>
      <View style={styles.linkList}>
        {links.map((link) => (
          <Pressable
            key={link.route}
            accessibilityRole="button"
            style={styles.linkCard}
            onPress={() => navigation.navigate(link.route)}
          >
            <View style={styles.linkIcon}>
              <Ionicons name={link.icon} size={22} color={colors.primary[700]} />
            </View>
            <View style={styles.linkCopy}>
              <Text style={styles.linkTitle}>{link.title}</Text>
              <Text style={styles.linkText}>{link.description}</Text>
            </View>
            <Ionicons name="chevron-forward" size={19} color={colors.gray[400]} />
          </Pressable>
        ))}
      </View>

      <View style={styles.guardianCard}>
        <Text style={styles.guardianEyebrow}>YOUR SITUATION CHANGED?</Text>
        <Text style={styles.guardianTitle}>Start dogOS with a dog you care for.</Text>
        <Text style={styles.guardianText}>
          Switching the view still does not create pet access. Woof will ask you to add or join a
          real relationship next.
        </Text>
        <Pressable
          accessibilityRole="button"
          disabled={switching}
          style={[styles.primaryButton, switching && styles.disabled]}
          onPress={() => void becomeGuardian()}
        >
          {switching ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.primaryButtonText}>I care for a dog now</Text>
          )}
        </Pressable>
      </View>

      {error && (
        <Text style={styles.errorText} accessibilityRole="alert">
          {error}
        </Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { paddingHorizontal: 20, paddingTop: 58, paddingBottom: 42 },
  heroIcon: {
    width: 50,
    height: 50,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[100],
    marginBottom: 24,
  },
  eyebrow: {
    color: colors.primary[700],
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.3,
    marginBottom: 8,
  },
  title: { color: colors.gray[900], fontSize: 31, lineHeight: 38, fontWeight: '800' },
  intro: { color: colors.gray[600], fontSize: 15, lineHeight: 23, marginTop: 12 },
  truthCard: {
    marginTop: 24,
    borderRadius: 18,
    backgroundColor: colors.success.light,
    padding: 16,
    flexDirection: 'row',
    gap: 11,
  },
  truthCopy: { flex: 1 },
  truthTitle: { color: colors.success.dark, fontSize: 14, fontWeight: '800' },
  truthText: { color: colors.success.dark, fontSize: 12, lineHeight: 18, marginTop: 4 },
  sectionTitle: {
    color: colors.gray[900],
    fontSize: 18,
    fontWeight: '800',
    marginTop: 30,
  },
  sectionIntro: {
    marginTop: 4,
    marginBottom: 12,
    color: colors.gray[600],
    fontSize: 12,
    lineHeight: 18,
  },
  linkList: { gap: 10 },
  linkCard: {
    minHeight: 88,
    borderWidth: 1,
    borderColor: colors.gray[200],
    borderRadius: 18,
    padding: 14,
    backgroundColor: '#ffffff',
    flexDirection: 'row',
    alignItems: 'center',
  },
  linkIcon: {
    width: 44,
    height: 44,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[50],
    marginRight: 12,
  },
  linkCopy: { flex: 1, paddingRight: 8 },
  linkTitle: { color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  linkText: { color: colors.gray[600], fontSize: 12, lineHeight: 18, marginTop: 3 },
  guardianCard: { marginTop: 28, borderRadius: 20, padding: 18, backgroundColor: colors.gray[900] },
  guardianEyebrow: {
    color: colors.primary[300],
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  guardianTitle: {
    color: '#ffffff',
    fontSize: 20,
    lineHeight: 26,
    fontWeight: '800',
    marginTop: 7,
  },
  guardianText: { color: colors.gray[300], fontSize: 13, lineHeight: 20, marginTop: 8 },
  primaryButton: {
    minHeight: 50,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    marginTop: 16,
  },
  primaryButtonText: { color: '#ffffff', fontSize: 14, fontWeight: '800' },
  disabled: { opacity: 0.55 },
  errorText: {
    color: colors.error.dark,
    backgroundColor: colors.error.light,
    borderRadius: 12,
    padding: 12,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 16,
  },
});
