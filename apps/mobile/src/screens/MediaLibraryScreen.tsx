import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { nativeMediaLibraryApi, type NativeMediaAsset } from '../api/media-library';
import { petsApi } from '../api/pets';
import { keepAppleMediaPrivately, pickApplePetMedia, sharePrivateMediaOnApple } from '../media/apple-media-adapter';
import type { Pet } from '../types';

export default function MediaLibraryScreen() {
  const [pets, setPets] = useState<Pet[]>([]);
  const [petId, setPetId] = useState('');
  const [assets, setAssets] = useState<NativeMediaAsset[]>([]);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);

  const selectedPet = useMemo(() => pets.find((pet) => pet.id === petId), [petId, pets]);

  const refresh = useCallback(async () => {
    if (!petId) {
      setAssets([]);
      return;
    }
    try {
      const result = await nativeMediaLibraryApi.library(petId);
      setAssets(result.assets);
    } catch {
      setAssets([]);
    }
  }, [petId]);

  useEffect(() => {
    void (async () => {
      try {
        const response = await petsApi.getPets();
        setPets(response.pets);
        if (response.pets[0]) setPetId(response.pets[0].id);
      } catch {
        Alert.alert('Private library unavailable', 'Woof could not load your pets.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function importFromApplePhotos() {
    if (!petId || working) return;
    setWorking(true);
    try {
      const picked = await pickApplePetMedia(20);
      if (!picked.length) return;
      await keepAppleMediaPrivately({ petId, items: picked, tags: ['apple photos'] });
      await refresh();
      Alert.alert('Saved privately', `${picked.length} selected ${picked.length === 1 ? 'item' : 'items'} added to ${selectedPet?.name ?? 'your pet'}'s library.`);
    } catch (error) {
      Alert.alert('Could not import', error instanceof Error ? error.message : 'Try again.');
    } finally {
      setWorking(false);
    }
  }

  async function share(asset: NativeMediaAsset) {
    try {
      await sharePrivateMediaOnApple(asset);
    } catch (error) {
      Alert.alert('Could not share', error instanceof Error ? error.message : 'Try again.');
    }
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.center}>
        <ActivityIndicator size="large" color="#7c5ce7" />
        <Text style={styles.muted}>Opening your private pet library…</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerCopy}>
          <Text style={styles.eyebrow}>PRIVATE PET LIBRARY</Text>
          <Text style={styles.title}>Moments that teach Woof</Text>
          <Text style={styles.subtitle}>Only media you explicitly keep is uploaded.</Text>
        </View>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Choose photos and videos from Apple Photos"
          onPress={importFromApplePhotos}
          disabled={!petId || working}
          style={({ pressed }) => [styles.addButton, pressed && styles.pressed, working && styles.disabled]}
        >
          {working ? <ActivityIndicator color="#fff" /> : <Ionicons name="images" size={20} color="#fff" />}
        </Pressable>
      </View>

      {pets.length > 1 && (
        <FlatList
          horizontal
          data={pets}
          keyExtractor={(item) => item.id}
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.petRow}
          renderItem={({ item }) => (
            <Pressable
              onPress={() => setPetId(item.id)}
              accessibilityRole="button"
              accessibilityState={{ selected: item.id === petId }}
              style={[styles.petPill, item.id === petId && styles.petPillSelected]}
            >
              <Text style={[styles.petPillText, item.id === petId && styles.petPillTextSelected]}>{item.name}</Text>
            </Pressable>
          )}
        />
      )}

      <FlatList
        data={assets}
        keyExtractor={(item) => item.id}
        numColumns={2}
        columnWrapperStyle={assets.length > 1 ? styles.gridRow : undefined}
        contentContainerStyle={assets.length ? styles.grid : styles.emptyContainer}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Ionicons name="images-outline" size={34} color="#8b8b96" />
            <Text style={styles.emptyTitle}>No private moments yet</Text>
            <Text style={styles.muted}>Choose a few photos or clips. Woof never scans your full Photos library.</Text>
            <Pressable onPress={importFromApplePhotos} style={styles.secondaryButton} accessibilityRole="button">
              <Text style={styles.secondaryButtonText}>Choose from Photos</Text>
            </Pressable>
          </View>
        }
        renderItem={({ item }) => {
          const displayUrl = item.thumbnailUrl || item.posterUrl || item.url;
          return (
            <Pressable
              style={styles.card}
              onLongPress={() => void share(item)}
              accessibilityRole="button"
              accessibilityLabel={`${item.filename}. Long press to share.`}
            >
              {displayUrl ? (
                <Image source={{ uri: displayUrl }} style={styles.media} resizeMode="cover" />
              ) : (
                <View style={[styles.media, styles.placeholder]}>
                  <Ionicons name={item.mediaType === 'video' ? 'videocam-outline' : 'image-outline'} size={28} color="#8b8b96" />
                </View>
              )}
              <View style={styles.cardFooter}>
                <Text numberOfLines={1} style={styles.filename}>{item.filename}</Text>
                <Pressable onPress={() => void share(item)} hitSlop={10} accessibilityLabel={`Share ${item.filename}`}>
                  <Ionicons name="share-outline" size={18} color="#5e5e69" />
                </Pressable>
              </View>
            </Pressable>
          );
        }}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fbfaf8' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, backgroundColor: '#fbfaf8' },
  header: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 12, flexDirection: 'row', alignItems: 'center', gap: 14 },
  headerCopy: { flex: 1 },
  eyebrow: { fontSize: 10, fontWeight: '700', letterSpacing: 1.6, color: '#7c5ce7' },
  title: { marginTop: 4, fontSize: 24, fontWeight: '800', color: '#232329' },
  subtitle: { marginTop: 5, fontSize: 13, lineHeight: 18, color: '#71717a' },
  addButton: { width: 48, height: 48, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: '#7c5ce7' },
  disabled: { opacity: 0.5 },
  pressed: { transform: [{ scale: 0.97 }] },
  petRow: { paddingHorizontal: 20, paddingBottom: 12, gap: 8 },
  petPill: { borderWidth: 1, borderColor: '#dedbe7', borderRadius: 999, paddingHorizontal: 14, paddingVertical: 9, backgroundColor: '#fff' },
  petPillSelected: { borderColor: '#7c5ce7', backgroundColor: '#f0ebff' },
  petPillText: { fontSize: 13, fontWeight: '600', color: '#67636f' },
  petPillTextSelected: { color: '#6742d8' },
  grid: { paddingHorizontal: 14, paddingBottom: 110 },
  gridRow: { gap: 10 },
  card: { flex: 1, margin: 5, overflow: 'hidden', borderRadius: 18, backgroundColor: '#fff', borderWidth: 1, borderColor: '#ece9f0' },
  media: { width: '100%', aspectRatio: 1 },
  placeholder: { alignItems: 'center', justifyContent: 'center', backgroundColor: '#f1eff3' },
  cardFooter: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 10, paddingVertical: 10 },
  filename: { flex: 1, fontSize: 12, color: '#4b4b52' },
  emptyContainer: { flexGrow: 1, paddingHorizontal: 28, paddingBottom: 100 },
  empty: { flex: 1, minHeight: 360, alignItems: 'center', justifyContent: 'center' },
  emptyTitle: { marginTop: 12, fontSize: 18, fontWeight: '700', color: '#2c2c32' },
  muted: { marginTop: 8, maxWidth: 300, textAlign: 'center', fontSize: 13, lineHeight: 19, color: '#777780' },
  secondaryButton: { marginTop: 18, borderRadius: 14, borderWidth: 1, borderColor: '#d8d3e3', backgroundColor: '#fff', paddingHorizontal: 16, paddingVertical: 12 },
  secondaryButtonText: { color: '#6742d8', fontSize: 13, fontWeight: '700' },
});
