import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { eventsApi } from '../api/events';
import { CommunityEvent } from '../types';
import { format } from 'date-fns';

export default function EventsScreen({ navigation }: any) {
  const [events, setEvents] = useState<CommunityEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await eventsApi.getEvents(1, 20);
      setEvents(response.data);
    } catch (error) {
      Alert.alert('Error', 'Failed to load events');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    loadEvents();
  };

  const handleRSVP = async (eventId: string, status: 'going' | 'maybe') => {
    try {
      await eventsApi.rsvpEvent(eventId, status);
      loadEvents();
      Alert.alert('Success', `RSVP updated to ${status}`);
    } catch (error) {
      Alert.alert('Error', 'Failed to RSVP');
    }
  };

  const renderEvent = ({ item }: { item: CommunityEvent }) => {
    const eventDate = new Date(item.startTime);
    const isUpcoming = eventDate > new Date();

    return (
      <TouchableOpacity
        style={styles.eventCard}
        onPress={() => navigation.navigate('EventDetail', { eventId: item.id })}
      >
        <View style={styles.eventDate}>
          <Text style={styles.eventMonth}>{format(eventDate, 'MMM')}</Text>
          <Text style={styles.eventDay}>{format(eventDate, 'd')}</Text>
        </View>

        <View style={styles.eventInfo}>
          <Text style={styles.eventTitle}>{item.title}</Text>
          <View style={styles.eventMeta}>
            <Ionicons name="time-outline" size={16} color="#6b7280" />
            <Text style={styles.eventMetaText}>
              {format(eventDate, 'h:mm a')}
            </Text>
          </View>
          <View style={styles.eventMeta}>
            <Ionicons name="location-outline" size={16} color="#6b7280" />
            <Text style={styles.eventMetaText} numberOfLines={1}>
              {item.location.address}
            </Text>
          </View>
          <View style={styles.eventMeta}>
            <Ionicons name="people-outline" size={16} color="#6b7280" />
            <Text style={styles.eventMetaText}>
              {item.rsvpsCount} going
            </Text>
          </View>
        </View>

        <View style={styles.eventActions}>
          {item.hasRsvped ? (
            <View style={styles.rsvpBadge}>
              <Ionicons name="checkmark-circle" size={20} color="#10b981" />
            </View>
          ) : isUpcoming ? (
            <TouchableOpacity
              style={styles.rsvpButton}
              onPress={() => handleRSVP(item.id, 'going')}
            >
              <Text style={styles.rsvpButtonText}>RSVP</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <Text>Loading events...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Events</Text>
        <TouchableOpacity onPress={() => navigation.navigate('CreateEvent')}>
          <Ionicons name="add-circle-outline" size={28} color="#8B5CF6" />
        </TouchableOpacity>
      </View>

      <FlatList
        data={events}
        renderItem={renderEvent}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyEmoji}>📅</Text>
            <Text style={styles.emptyText}>No events yet</Text>
            <Text style={styles.emptySubtext}>Create the first event!</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  listContent: {
    padding: 16,
  },
  eventCard: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  eventDate: {
    backgroundColor: '#8B5CF6',
    borderRadius: 8,
    width: 60,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
  },
  eventMonth: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  eventDay: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  eventInfo: {
    flex: 1,
    marginLeft: 16,
  },
  eventTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  eventMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  eventMetaText: {
    marginLeft: 6,
    fontSize: 14,
    color: '#6b7280',
    flex: 1,
  },
  eventActions: {
    justifyContent: 'center',
    marginLeft: 8,
  },
  rsvpButton: {
    backgroundColor: '#8B5CF6',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  rsvpButtonText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  rsvpBadge: {
    alignItems: 'center',
  },
  emptyContainer: {
    alignItems: 'center',
    padding: 48,
  },
  emptyEmoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#6b7280',
  },
});
