import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useAuth } from '../contexts/AuthContext';

export default function HomeScreen() {
  const { user, logout } = useAuth();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to Woof!</Text>
      <Text style={styles.subtitle}>Hello, {user?.displayName}</Text>
      <Text style={styles.handle}>@{user?.handle}</Text>

      <View style={styles.content}>
        <Text style={styles.emoji}>🐾</Text>
        <Text style={styles.message}>
          Your mobile app is successfully connected to the API!
        </Text>
        <Text style={styles.info}>
          Next steps: Add features like pet profiles, map view, events, and more.
        </Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={logout}>
        <Text style={styles.buttonText}>Log Out</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
    padding: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 20,
    color: '#6b7280',
    marginBottom: 4,
  },
  handle: {
    fontSize: 16,
    color: '#8B5CF6',
    marginBottom: 32,
  },
  content: {
    alignItems: 'center',
    marginBottom: 48,
  },
  emoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  message: {
    fontSize: 18,
    color: '#1f2937',
    textAlign: 'center',
    marginBottom: 16,
  },
  info: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    maxWidth: 300,
  },
  button: {
    backgroundColor: '#ef4444',
    borderRadius: 8,
    padding: 16,
    minWidth: 200,
    alignItems: 'center',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
});
