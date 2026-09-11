import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator, type StackScreenProps } from '@react-navigation/stack';
import { companionApi, type CompanionState } from '../api/companion';
import { useAuth } from '../contexts/AuthContext';
import { clearPetCreationRecovery } from '../onboarding/recovery';
import { colors } from '../theme/tokens';

// Auth screens
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';

// dogOS primary loop
import TodayScreen from '../screens/TodayScreen';
import CompassScreen from '../screens/CompassScreen';
import StoryScreen from '../screens/StoryScreen';
import FeedScreen from '../screens/FeedScreen';

// Onboarding / Companion entry
import CompanionModeScreen from '../screens/CompanionModeScreen';
import FirstAdventureScreen from '../screens/FirstAdventureScreen';
import CompanionHomeScreen from '../screens/CompanionHomeScreen';

// Contextual / secondary tools
import DailySignalsScreen from '../screens/DailySignalsScreen';
import EventsScreen from '../screens/EventsScreen';
import MapScreen from '../screens/MapScreen';
import PetsListScreen from '../screens/PetsListScreen';
import GoalsScreen from '../screens/GoalsScreen';
import MediaLibraryScreen from '../screens/MediaLibraryScreen';
import ProfileScreen from '../screens/ProfileScreen';
import SkillcraftScreen from '../screens/SkillcraftScreen';
import PacksScreen from '../screens/PacksScreen';
import ExpeditionScreen from '../screens/ExpeditionScreen';

export type RootStackParamList = {
  Login: undefined;
  Register: undefined;
  MainTabs: undefined;
  CompanionHome: undefined;
  CommunityStandalone: undefined;
  DailySignals: undefined;
  Events: undefined;
  Map: undefined;
  Pets: undefined;
  Goals: undefined;
  Library: undefined;
  Profile: undefined;
  Skillcraft: undefined;
  Packs: undefined;
  Expedition: undefined;
};

export type MainTabParamList = {
  Today: undefined;
  Compass: undefined;
  Story: undefined;
  Community: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();
const StandaloneFeedScreen = FeedScreen as unknown as React.ComponentType<
  StackScreenProps<RootStackParamList, 'CommunityStandalone'>
>;

const tabIcons: Record<
  keyof MainTabParamList,
  { active: keyof typeof Ionicons.glyphMap; inactive: keyof typeof Ionicons.glyphMap }
> = {
  Today: { active: 'paw', inactive: 'paw-outline' },
  Compass: { active: 'compass', inactive: 'compass-outline' },
  Story: { active: 'book', inactive: 'book-outline' },
  Community: { active: 'people', inactive: 'people-outline' },
};

const MainTabs = () => (
  <Tab.Navigator
    screenOptions={({ route }) => ({
      tabBarIcon: ({ focused, color, size }) => (
        <Ionicons
          name={focused ? tabIcons[route.name].active : tabIcons[route.name].inactive}
          size={size}
          color={color}
        />
      ),
      tabBarActiveTintColor: '#7c3aed',
      tabBarInactiveTintColor: '#6b7280',
      headerShown: false,
      tabBarLabelStyle: { fontSize: 11, fontWeight: '600' },
      tabBarStyle: {
        backgroundColor: '#ffffff',
        borderTopWidth: StyleSheet.hairlineWidth,
        borderTopColor: '#e5e7eb',
        paddingBottom: 6,
        paddingTop: 6,
        height: 66,
      },
    })}
  >
    <Tab.Screen name="Today" component={TodayScreen} />
    <Tab.Screen name="Compass" component={CompassScreen} />
    <Tab.Screen name="Story" component={StoryScreen} />
    <Tab.Screen name="Community" component={FeedScreen} />
  </Tab.Navigator>
);

const secondaryScreenOptions = {
  headerBackTitle: 'Back',
  headerTintColor: '#6d28d9',
  headerTitleStyle: { fontWeight: '700' as const },
  headerStyle: { backgroundColor: '#ffffff' },
  cardStyle: { backgroundColor: '#f9fafb' },
};

function AuthNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ cardStyle: { backgroundColor: '#ffffff' } }}>
        <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Register" component={RegisterScreen} options={{ headerShown: false }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

function GuardianNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ cardStyle: { backgroundColor: '#ffffff' } }}>
        <Stack.Screen name="MainTabs" component={MainTabs} options={{ headerShown: false }} />
        <Stack.Screen
          name="DailySignals"
          component={DailySignalsScreen}
          options={{ ...secondaryScreenOptions, title: 'Daily Signals' }}
        />
        <Stack.Screen
          name="Pets"
          component={PetsListScreen}
          options={{ ...secondaryScreenOptions, title: 'Pets' }}
        />
        <Stack.Screen
          name="Goals"
          component={GoalsScreen}
          options={{ ...secondaryScreenOptions, title: 'Goals' }}
        />
        <Stack.Screen
          name="Library"
          component={MediaLibraryScreen}
          options={{ ...secondaryScreenOptions, title: 'Library' }}
        />
        <Stack.Screen
          name="Events"
          component={EventsScreen}
          options={{ ...secondaryScreenOptions, title: 'Events' }}
        />
        <Stack.Screen
          name="Map"
          component={MapScreen}
          options={{ ...secondaryScreenOptions, title: 'Map' }}
        />
        <Stack.Screen
          name="Skillcraft"
          component={SkillcraftScreen}
          options={{ ...secondaryScreenOptions, title: 'Skillcraft' }}
        />
        <Stack.Screen
          name="Packs"
          component={PacksScreen}
          options={{ ...secondaryScreenOptions, title: 'Packs' }}
        />
        <Stack.Screen
          name="Expedition"
          component={ExpeditionScreen}
          options={{ ...secondaryScreenOptions, title: 'Expedition' }}
        />
        <Stack.Screen
          name="Profile"
          component={ProfileScreen}
          options={{ ...secondaryScreenOptions, title: 'You' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

function CompanionNavigator({ onResolved }: { onResolved: (state: CompanionState) => void }) {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ cardStyle: { backgroundColor: '#ffffff' } }}>
        <Stack.Screen name="CompanionHome" options={{ headerShown: false }}>
          {(props) => <CompanionHomeScreen {...props} onResolved={onResolved} />}
        </Stack.Screen>
        <Stack.Screen
          name="CommunityStandalone"
          component={StandaloneFeedScreen}
          options={{ ...secondaryScreenOptions, title: 'Community' }}
        />
        <Stack.Screen
          name="Events"
          component={EventsScreen}
          options={{ ...secondaryScreenOptions, title: 'Events' }}
        />
        <Stack.Screen
          name="Map"
          component={MapScreen}
          options={{ ...secondaryScreenOptions, title: 'Map' }}
        />
        <Stack.Screen
          name="Skillcraft"
          component={SkillcraftScreen}
          options={{ ...secondaryScreenOptions, title: 'Skillcraft' }}
        />
        <Stack.Screen
          name="Packs"
          component={PacksScreen}
          options={{ ...secondaryScreenOptions, title: 'Packs' }}
        />
        <Stack.Screen
          name="Expedition"
          component={ExpeditionScreen}
          options={{ ...secondaryScreenOptions, title: 'Expedition' }}
        />
        <Stack.Screen
          name="Profile"
          component={ProfileScreen}
          options={{ ...secondaryScreenOptions, title: 'You' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

function AuthenticatedEntry() {
  const { logout } = useAuth();
  const [state, setState] = useState<CompanionState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const applyResolved = useCallback(async (next: CompanionState) => {
    if (next.landing === 'PET_TODAY') {
      try {
        await clearPetCreationRecovery();
      } catch {
        // Server pet authority is already canonical. Stale local retry metadata
        // is never allowed to block entry and will be overwritten if needed.
      }
    }
    setState(next);
    setError(null);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const next = await companionApi.state();
      await applyResolved(next);
    } catch {
      setState(null);
      setError(
        'Woof could not verify your account mode. Pet-specific surfaces stay closed until server authority is available.'
      );
    } finally {
      setLoading(false);
    }
  }, [applyResolved]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading && !state) {
    return (
      <View style={styles.loadingContainer} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening the right Woof for you…</Text>
      </View>
    );
  }

  if (error || !state) {
    return (
      <View style={styles.authorityError}>
        <Ionicons name="shield-outline" size={34} color={colors.primary[700]} />
        <Text style={styles.errorTitle}>Woof mode needs server verification.</Text>
        <Text style={styles.errorCopy}>{error}</Text>
        <Pressable
          accessibilityRole="button"
          style={styles.retryButton}
          onPress={() => void load()}
        >
          <Text style={styles.retryButtonText}>Try again</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          style={styles.signOutButton}
          onPress={() => void logout()}
        >
          <Text style={styles.signOutText}>Sign out</Text>
        </Pressable>
      </View>
    );
  }

  if (state.landing === 'NEEDS_MODE') {
    return <CompanionModeScreen onResolved={(next) => void applyResolved(next)} />;
  }

  if (state.landing === 'NEEDS_PET_SETUP') {
    return (
      <FirstAdventureScreen
        onComplete={() => void load()}
        onModeResolved={(next) => void applyResolved(next)}
        onRecheck={() => void load()}
      />
    );
  }

  if (state.landing === 'COMPANION_TODAY') {
    return <CompanionNavigator onResolved={(next) => void applyResolved(next)} />;
  }

  if (state.landing === 'PET_TODAY') {
    return <GuardianNavigator />;
  }

  return (
    <View style={styles.authorityError}>
      <Ionicons name="shield-outline" size={34} color={colors.primary[700]} />
      <Text style={styles.errorTitle}>Woof returned an unsupported account mode.</Text>
      <Text style={styles.errorCopy}>
        Pet-specific surfaces stay closed until this client can verify a recognized server landing.
      </Text>
      <Pressable accessibilityRole="button" style={styles.retryButton} onPress={() => void load()}>
        <Text style={styles.retryButtonText}>Check again</Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        style={styles.signOutButton}
        onPress={() => void logout()}
      >
        <Text style={styles.signOutText}>Sign out</Text>
      </Pressable>
    </View>
  );
}

function SessionVerificationGate() {
  const { logout, retrySessionVerification } = useAuth();

  return (
    <View style={styles.authorityError}>
      <Ionicons name="cloud-offline-outline" size={34} color={colors.primary[700]} />
      <Text style={styles.errorTitle}>Woof can’t verify this session yet.</Text>
      <Text style={styles.errorCopy}>
        Your saved sign-in is still on this device, but authenticated Woof surfaces stay closed
        until the server can confirm it. Try again, or sign out on this device.
      </Text>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Retry Woof session verification"
        style={styles.retryButton}
        onPress={() => void retrySessionVerification()}
      >
        <Text style={styles.retryButtonText}>Try again</Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Sign out on this device"
        style={styles.signOutButton}
        onPress={() => void logout()}
      >
        <Text style={styles.signOutText}>Sign out on this device</Text>
      </Pressable>
    </View>
  );
}

export const AppNavigator = () => {
  const { isAuthenticated, loading, sessionVerificationUnavailable } = useAuth();

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary[600]} />
      </View>
    );
  }

  if (sessionVerificationUnavailable) {
    return <SessionVerificationGate />;
  }

  return isAuthenticated ? <AuthenticatedEntry /> : <AuthNavigator />;
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    padding: 24,
  },
  loadingText: { color: colors.gray[600], fontSize: 14, marginTop: 12 },
  authorityError: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    paddingHorizontal: 28,
  },
  errorTitle: {
    color: colors.gray[900],
    fontSize: 21,
    fontWeight: '800',
    textAlign: 'center',
    marginTop: 14,
  },
  errorCopy: {
    color: colors.gray[600],
    fontSize: 14,
    lineHeight: 21,
    textAlign: 'center',
    marginTop: 8,
  },
  retryButton: {
    minHeight: 48,
    minWidth: 160,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    marginTop: 22,
  },
  retryButtonText: { color: '#ffffff', fontSize: 14, fontWeight: '800' },
  signOutButton: { paddingHorizontal: 24, paddingVertical: 14, marginTop: 4 },
  signOutText: { color: colors.gray[600], fontSize: 13, fontWeight: '700' },
});
