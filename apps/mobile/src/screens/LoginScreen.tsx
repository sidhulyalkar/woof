import React, { useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { useAuth } from '../contexts/AuthContext';
import { RootStackParamList } from '../navigation/AppNavigator';

type LoginScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Login'>;

interface Props {
  navigation: LoginScreenNavigationProp;
}

export default function LoginScreen({ navigation }: Props) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login, loading } = useAuth();

  const handleLogin = async () => {
    if (!email.trim() || !password) {
      Alert.alert('Check your details', 'Enter your email and password.');
      return;
    }

    try {
      await login(email.trim(), password);
    } catch (error: any) {
      const message = error.response?.data?.message || 'Invalid email or password';
      Alert.alert('Sign in failed', Array.isArray(message) ? message[0] : message);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
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
          <Text style={styles.eyebrow}>CONTEXT → CONFIDENCE → COORDINATION</Text>
          <Text style={styles.title}>Better dog friendships, offline.</Text>
          <Text style={styles.subtitle}>
            Sign in to discover explainable matches, coordinate meetups, and keep the real-world outcome loop moving.
          </Text>
        </View>

        <View style={styles.form}>
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
            placeholder="Your password"
            placeholderTextColor="#697480"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            textContentType="password"
            editable={!loading}
            accessibilityLabel="Password"
            onSubmitEditing={() => void handleLogin()}
          />

          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={handleLogin}
            disabled={loading}
            accessibilityRole="button"
            accessibilityLabel="Sign in to Woof"
          >
            <Text style={styles.buttonText}>{loading ? 'Signing in…' : 'Sign in'}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => navigation.navigate('Register')}
            disabled={loading}
            accessibilityRole="button"
          >
            <Text style={styles.linkText}>
              New to Woof? <Text style={styles.linkTextBold}>Create an account</Text>
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.footerCard}>
          <Text style={styles.footerTitle}>Designed for real-world coordination</Text>
          <Text style={styles.footerText}>
            Location is requested in context, model scores stay explainable, and optional services degrade without blocking your core profile.
          </Text>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D1117' },
  content: { flex: 1, justifyContent: 'center', paddingHorizontal: 24, paddingVertical: 44 },
  brandRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 48 },
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
  hero: { marginBottom: 34 },
  eyebrow: { color: '#55D6BE', fontSize: 10, fontWeight: '800', letterSpacing: 1.3, marginBottom: 10 },
  title: { color: '#F5F3EE', fontSize: 36, lineHeight: 41, fontWeight: '800', letterSpacing: -0.7 },
  subtitle: { color: '#A8B0BA', fontSize: 15, lineHeight: 23, marginTop: 13 },
  form: { width: '100%' },
  label: { color: '#D7DBE0', fontSize: 13, fontWeight: '700', marginBottom: 8 },
  input: {
    minHeight: 54,
    borderWidth: 1,
    borderColor: '#2A3541',
    borderRadius: 14,
    paddingHorizontal: 16,
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
  footerCard: {
    borderTopWidth: 1,
    borderTopColor: '#27313C',
    marginTop: 32,
    paddingTop: 20,
  },
  footerTitle: { color: '#55D6BE', fontSize: 12, fontWeight: '800', letterSpacing: 0.3 },
  footerText: { color: '#7F8994', fontSize: 12, lineHeight: 18, marginTop: 6 },
});
