import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Alert, TouchableOpacity } from 'react-native';
import MapView, { Marker, PROVIDER_GOOGLE } from 'react-native-maps';
import * as Location from 'expo-location';
import { Ionicons } from '@expo/vector-icons';
import { petsApi } from '../api/pets';
import { Pet } from '../types';

export default function MapScreen({ navigation }: any) {
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [nearbyPets, setNearbyPets] = useState<Pet[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    requestLocationPermission();
  }, []);

  const requestLocationPermission = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permission is required to show nearby pets');
        setLoading(false);
        return;
      }

      const currentLocation = await Location.getCurrentPositionAsync({});
      setLocation(currentLocation);
      loadNearbyPets(currentLocation.coords.latitude, currentLocation.coords.longitude);
    } catch (error) {
      Alert.alert('Error', 'Failed to get location');
      setLoading(false);
    }
  };

  const loadNearbyPets = async (latitude: number, longitude: number) => {
    try {
      const pets = await petsApi.getNearbyPets(latitude, longitude, 5000);
      setNearbyPets(pets);
    } catch (error) {
      Alert.alert('Error', 'Failed to load nearby pets');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    if (location) {
      loadNearbyPets(location.coords.latitude, location.coords.longitude);
    }
  };

  if (loading || !location) {
    return (
      <View style={styles.centerContainer}>
        <Text>Loading map...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        provider={PROVIDER_GOOGLE}
        style={styles.map}
        initialRegion={{
          latitude: location.coords.latitude,
          longitude: location.coords.longitude,
          latitudeDelta: 0.05,
          longitudeDelta: 0.05,
        }}
        showsUserLocation
        showsMyLocationButton
      >
        {nearbyPets.map((pet) => (
          <Marker
            key={pet.id}
            coordinate={{
              latitude: pet.owner?.location ? parseFloat(pet.owner.location.split(',')[0]) : 0,
              longitude: pet.owner?.location ? parseFloat(pet.owner.location.split(',')[1]) : 0,
            }}
            title={pet.name}
            description={`${pet.breed || pet.species} • ${pet.age} years old`}
          >
            <View style={styles.marker}>
              <Text style={styles.markerEmoji}>🐾</Text>
            </View>
          </Marker>
        ))}
      </MapView>

      <View style={styles.header}>
        <Text style={styles.headerTitle}>Nearby Pets</Text>
        <TouchableOpacity onPress={handleRefresh}>
          <Ionicons name="refresh" size={24} color="#8B5CF6" />
        </TouchableOpacity>
      </View>

      <View style={styles.statsCard}>
        <Text style={styles.statsText}>
          {nearbyPets.length} pets within 5km
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  map: {
    flex: 1,
  },
  header: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  statsCard: {
    position: 'absolute',
    bottom: 32,
    left: 16,
    right: 16,
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  statsText: {
    fontSize: 16,
    color: '#1f2937',
    fontWeight: '600',
  },
  marker: {
    backgroundColor: '#8B5CF6',
    borderRadius: 20,
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#ffffff',
  },
  markerEmoji: {
    fontSize: 20,
  },
});
