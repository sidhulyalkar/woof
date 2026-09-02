import React, { useCallback, useEffect, useState } from 'react';
import { Alert, Image, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { StackScreenProps } from '@react-navigation/stack';
import { petsApi } from '../api/pets';
import { useAuth } from '../contexts/AuthContext';
import type { RootStackParamList } from '../navigation/AppNavigator';
import type { Pet } from '../types';

type Props = StackScreenProps<RootStackParamList, 'Profile'>;

export default function ProfileScreen({ navigation }: Props) {
  const { user, logout } = useAuth();
  const [pets, setPets] = useState<Pet[]>([]);
  const [petsLoading, setPetsLoading] = useState(true);
  const [petsUnavailable, setPetsUnavailable] = useState(false);

  const loadPets = useCallback(async () => {
    if (!user?.id) {
      setPets([]);
      setPetsUnavailable(false);
      setPetsLoading(false);
      return;
    }

    setPetsLoading(true);
    setPetsUnavailable(false);
    try {
      const petsData = await petsApi.getPets(user.id);
      setPets(petsData.pets);
    } catch {
      setPets([]);
      setPetsUnavailable(true);
    } finally {
      setPetsLoading(false);
    }
  }, [user?.id]);

  useEffect(() => {
    void loadPets();
  }, [loadPets]);

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        style: 'destructive',
        onPress: () => void logout(),
      },
    ]);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.profileSection}>
        {user?.avatarUrl ? (
          <Image source={{ uri: user.avatarUrl }} style={styles.avatar} />
        ) : (
          <View style={styles.avatarFallback}>
            <Ionicons name="person-outline" size={38} color="#6b7280" />
          </View>
        )}
        <Text style={styles.displayName}>{user?.displayName || user?.handle || 'Woof member'}</Text>
        {user?.handle ? <Text style={styles.handle}>@{user.handle}</Text> : null}
        {user?.bio ? <Text style={styles.bio}>{user.bio}</Text> : null}
      </View>

      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <View>
            <Text style={styles.eyebrow}>The pack</Text>
            <Text style={styles.sectionTitle}>My pets</Text>
          </View>
          <TouchableOpacity onPress={() => navigation.navigate('Pets')}>
            <Text style={styles.seeAllText}>Open Pets</Text>
          </TouchableOpacity>
        </View>

        {petsLoading ? (
          <Text style={styles.mutedText}>Loading your pets...</Text>
        ) : petsUnavailable ? (
          <View style={styles.noticeCard}>
            <Text style={styles.noticeTitle}>Pets are temporarily unavailable</Text>
            <Text style={styles.mutedText}>
              Your account profile is still available. You can retry from Pets.
            </Text>
          </View>
        ) : pets.length === 0 ? (
          <View style={styles.noticeCard}>
            <Text style={styles.noticeTitle}>No pet profile yet</Text>
            <Text style={styles.mutedText}>
              Add your first pet to unlock dog-specific experiences.
            </Text>
          </View>
        ) : (
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {pets.map((pet) => (
              <TouchableOpacity
                key={pet.id}
                style={styles.petCard}
                onPress={() => navigation.navigate('Pets')}
              >
                {pet.avatarUrl ? (
                  <Image source={{ uri: pet.avatarUrl }} style={styles.petImage} />
                ) : (
                  <View style={styles.petImageFallback}>
                    <Ionicons name="paw-outline" size={28} color="#7c3aed" />
                  </View>
                )}
                <Text style={styles.petName} numberOfLines={1}>
                  {pet.name}
                </Text>
                <Text style={styles.petMeta} numberOfLines={1}>
                  {pet.breed || pet.species || 'Pet'}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.eyebrow}>dogOS</Text>
        <Text style={styles.sectionTitle}>Relationship tools</Text>

        <TouchableOpacity style={styles.menuItem} onPress={() => navigation.navigate('DailySignals')}>
          <Ionicons name="pulse-outline" size={24} color="#6b7280" />
          <View style={styles.menuCopy}>
            <Text style={styles.menuItemText}>Daily Signals</Text>
            <Text style={styles.menuItemDetail}>A quick private check-in for longitudinal context.</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#d1d5db" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.menuItem} onPress={() => navigation.navigate('Goals')}>
          <Ionicons name="flag-outline" size={24} color="#6b7280" />
          <View style={styles.menuCopy}>
            <Text style={styles.menuItemText}>Goals</Text>
            <Text style={styles.menuItemDetail}>Shared intentions and activity planning.</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#d1d5db" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.menuItem} onPress={() => navigation.navigate('Events')}>
          <Ionicons name="calendar-outline" size={24} color="#6b7280" />
          <View style={styles.menuCopy}>
            <Text style={styles.menuItemText}>Community events</Text>
            <Text style={styles.menuItemDetail}>Browse and manage event participation.</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#d1d5db" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.menuItem} onPress={() => navigation.navigate('Library')}>
          <Ionicons name="images-outline" size={24} color="#6b7280" />
          <View style={styles.menuCopy}>
            <Text style={styles.menuItemText}>Media library</Text>
            <Text style={styles.menuItemDetail}>Open your private Woof media.</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#d1d5db" />
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <TouchableOpacity style={styles.logoutItem} onPress={handleLogout}>
          <Ionicons name="log-out-outline" size={24} color="#ef4444" />
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f9fafb' },
  contentContainer: { paddingBottom: 32 },
  profileSection: { alignItems: 'center', paddingHorizontal: 24, paddingVertical: 32 },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#e5e7eb',
    marginBottom: 16,
  },
  avatarFallback: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e5e7eb',
    marginBottom: 16,
  },
  displayName: { fontSize: 24, fontWeight: 'bold', color: '#1f2937', marginBottom: 4 },
  handle: { fontSize: 16, color: '#6b7280', marginBottom: 8 },
  bio: { fontSize: 14, color: '#6b7280', textAlign: 'center', marginTop: 4 },
  section: { backgroundColor: '#ffffff', padding: 16, marginBottom: 8 },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    marginBottom: 16,
  },
  eyebrow: {
    fontSize: 11,
    fontWeight: '700',
    color: '#8B5CF6',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  sectionTitle: { fontSize: 18, fontWeight: '600', color: '#1f2937' },
  seeAllText: { fontSize: 14, color: '#8B5CF6', fontWeight: '600' },
  mutedText: { fontSize: 13, lineHeight: 19, color: '#6b7280' },
  noticeCard: { padding: 14, borderRadius: 12, backgroundColor: '#f9fafb' },
  noticeTitle: { fontSize: 14, fontWeight: '600', color: '#1f2937', marginBottom: 4 },
  petCard: { marginRight: 16, width: 92 },
  petImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#e5e7eb',
    marginBottom: 8,
  },
  petImageFallback: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ede9fe',
    marginBottom: 8,
  },
  petName: { fontSize: 13, fontWeight: '600', color: '#1f2937' },
  petMeta: { fontSize: 11, color: '#6b7280', marginTop: 2 },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  menuCopy: { flex: 1, marginLeft: 14, marginRight: 8 },
  menuItemText: { fontSize: 16, color: '#1f2937', fontWeight: '500' },
  menuItemDetail: { fontSize: 12, lineHeight: 17, color: '#6b7280', marginTop: 2 },
  logoutItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8 },
  logoutText: { marginLeft: 14, fontSize: 16, color: '#ef4444', fontWeight: '500' },
});
