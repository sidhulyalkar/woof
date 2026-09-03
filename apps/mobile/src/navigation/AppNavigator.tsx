import React from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { useAuth } from '../contexts/AuthContext';

// Auth screens
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';

// dogOS primary loop
import TodayScreen from '../screens/TodayScreen';
import CompassScreen from '../screens/CompassScreen';
import StoryScreen from '../screens/StoryScreen';
import FeedScreen from '../screens/FeedScreen';

// Contextual / secondary tools
import DailySignalsScreen from '../screens/DailySignalsScreen';
import EventsScreen from '../screens/EventsScreen';
import MapScreen from '../screens/MapScreen';
import PetsListScreen from '../screens/PetsListScreen';
import GoalsScreen from '../screens/GoalsScreen';
import MediaLibraryScreen from '../screens/MediaLibraryScreen';
import ProfileScreen from '../screens/ProfileScreen';

export type RootStackParamList = {
  Login: undefined;
  Register: undefined;
  MainTabs: undefined;
  DailySignals: undefined;
  Events: undefined;
  Map: undefined;
  Pets: undefined;
  Goals: undefined;
  Library: undefined;
  Profile: undefined;
};

export type MainTabParamList = {
  Today: undefined;
  Compass: undefined;
  Story: undefined;
  Community: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

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

export const AppNavigator = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#7c3aed" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ cardStyle: { backgroundColor: '#ffffff' } }}>
        {!isAuthenticated ? (
          <>
            <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
            <Stack.Screen
              name="Register"
              component={RegisterScreen}
              options={{ headerShown: false }}
            />
          </>
        ) : (
          <>
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
              name="Profile"
              component={ProfileScreen}
              options={{ ...secondaryScreenOptions, title: 'You' }}
            />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#ffffff',
  },
});
