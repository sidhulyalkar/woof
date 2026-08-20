import React, { useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { useAuth } from '../contexts/AuthContext';
import { RootStackParamList } from '../navigation/AppNavigator';

type RegisterScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Register'>;

interface Props {
  navigation: RegisterScreenNavigationProp;
}

export default function RegisterScreen({ navigation }: Props) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [handle, setHandle] = useState('');
  const { register, loading } = useAuth();

  const handleRegister = async () => {
    const normalizedHandle = handle.trim().toLowerCase().replace(/\s+/g, '_');

    if (!email.trim() || !password || normalizedHandle.length < 3) {
      Alert.alert('Check your details', 'Enter a valid email, a handle with at least 3 characters, and a password.');
      return;
    }

    if (password.length < 8) {
      Alert.alert('Password too short', 'Use at least 8 characters.');
      return;
    }

    try {
      await register(email.trim(), password, normalizedHandle);
    } catch (error: any) {
      const message = error.response?.data?.message || 'Unable to create account';
      Alert.alert('Registration failed', Array.isArray(message) ? message[0] : message);
    }
  };

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        <View style={styles.content}>
          <View style={styles.brandRow} accessibilityRole="header">
            <View style={styles.brandMark}>
              <Text style={styles.brandEmoji}>🐾</Text>
            </View>
            <View>
              <Text style={styles.brandEyebrow}>YOUR LOCAL PACK</Text>
              <Text style={styles.brandName}>Woof</Text>
            </View>
          </View>

          <View style={styles.hero}>
            <Text style={styles.eyebrow}>ACCOUNT FIRST, CONTEXT LATER</Text>
            <Text style={styles.title}>Create only what Woof needs to sign you in.</Text>
            <Text style={styles.subtitle}>
              Pet details, matching preferences, and location permissions should be requested in context after the account exists.
            </Text>
          </View>

          <View style={styles.privacyCard}>
            <Text style={styles.privacyIcon}>🛡️</Text>
            <View style={styles.privacyCopy}>
              <Text style={styles.privacyTitle}>Data minimization by default</Text>
              <Text style={styles.privacyText}>No home location, age, or route history is required to register.</Text>
            </View>
          </View>

          <View style={styles.form}>
            <Text style={styles.label}>Public handle</Text>
            <TextInput
              style={styles.input}
              placeholder="trailpaws"
              placeholderTextColor="#697480"
              value={handle}
              onChangeText={setHandle}
              autoCapitalize="none"
              autoCorrect={false}
              textContentType="username"
              editable={!loading}
              accessibilityLabel="Public handle"
              maxLength={30}
            />

            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="you@example.com"
              placeholderTextColor="#697480"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
              textContentType="emailAddress"
              editable={!loading}
              accessibilityLabel="Email"
            />

            <Text style={styles.label}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="At least 8 characters"
              placeholderTextColor="#697480"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              textContentType="newPassword"
              editable={!loading}
              accessibilityLabel="Password"
            />

            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={handleRegister}
              disabled={loading}
              accessibilityRole="button"
              accessibilityLabel="Create Woof account"
            >
              <Text style={styles.buttonText}>{loading ? 'Creating account…' : 'Create account'}</Text>
            </TouchableOpacity>

            <TouchableOpacity onPress={() => navigation.navigate('Login')} disabled={loading} accessibilityRole="button">
              <Text style={styles.linkText}>
                Already have an account? <Text style={styles.linkTextBold}>Sign in</Text>
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D1117' },
  scrollContent: { flexGrow: 1 },
  content: { flex: 1, paddingHorizontal: 24, paddingTop: 64, paddingBottom: 36 },
  brandRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 44 },
  brandMark: {
    width: 46,
    height: 46,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFB454',
    marginRight: 12,
  },
  brandEmoji: { fontSize: 23 },
  brandEyebrow: { color: '#89939E', fontSize: 10, fontWeight: '700', letterSpacing: 1.8 },
  brandName: { color: '#F5F3EE', fontSize: 20, fontWeight: '800', marginTop: 2 },
  hero: { marginBottom: 24 },
  eyebrow: { color: '#FFB454', fontSize: 11, fontWeight: '800', letterSpacing: 1.4, marginBottom: 10 },
  title: { color: '#F5F3EE', fontSize: 32, lineHeight: 38, fontWeight: '800', letterSpacing: -0.6 },
  subtitle: { color: '#A8B0BA', fontSize: 15, lineHeight: 23, marginTop: 12 },
  privacyCard: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: '#27313C',
    backgroundColor: '#141B22',
    borderRadius: 18,
    padding: 16,
    marginBottom: 28,
  },
  privacyIcon: { fontSize: 22, marginRight: 12 },
  privacyCopy: { flex: 1 },
  privacyTitle: { color: '#55D6BE', fontSize: 14, fontWeight: '700' },
  privacyText: { color: '#8F99A5', fontSize: 12, lineHeight: 18, marginTop: 4 },
  form: { width: '100%' },
  label: { color: '#D7DBE0', fontSize: 13, fontWeight: '700', marginBottom: 8 },
  input: {
    borderWidth: 1,
    borderColor: '#2A3541',
    borderRadius: 14,
    paddingHorizontal: 16,
    minHeight: 54,
    marginBottom: 18,
    fontSize: 16,
    color: '#F5F3EE',
    backgroundColor: '#151B22',
  },
  button: {
    minHeight: 54,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFB454',
    marginTop: 4,
    marginBottom: 18,
  },
  buttonDisabled: { opacity: 0.55 },
  buttonText: { color: '#24170A', fontSize: 16, fontWeight: '800' },
  linkText: { textAlign: 'center', color: '#8F99A5', fontSize: 14, paddingVertical: 10 },
  linkTextBold: { color: '#FFB454', fontWeight: '800' },
});
