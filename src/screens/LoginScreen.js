import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [confirmPassword, setConfirmPassword] = useState('');

  const handleLogin = async () => {
    if (!email.trim() || !password) {
      Alert.alert('Validation', 'Please enter email and password');
      return;
    }

    try {
      // NOTE: This is a simple local auth placeholder.
      // Replace with real auth calls for production.
      await AsyncStorage.setItem('userToken', JSON.stringify({ email }));
      navigation.reset({ index: 0, routes: [{ name: 'Dashboard' }] });
    } catch (e) {
      Alert.alert('Error', 'Failed to save login state');
    }
  };

  const handleSignUp = async () => {
    if (!email.trim() || !password) {
      Alert.alert('Validation', 'Please enter email and password');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Validation', 'Passwords do not match');
      return;
    }

    if (password.length < 6) {
      Alert.alert('Validation', 'Password must be at least 6 characters');
      return;
    }

    try {
      // NOTE: This is a simple local auth placeholder.
      // Replace with real auth calls for production.
      await AsyncStorage.setItem('userToken', JSON.stringify({ email }));
      navigation.reset({ index: 0, routes: [{ name: 'Dashboard' }] });
    } catch (e) {
      Alert.alert('Error', 'Failed to create account');
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      // NOTE: Implement Google Sign-In using @react-native-google-signin
      // For now, this is a placeholder
      Alert.alert('Coming Soon', 'Google Sign-In will be implemented soon');
    } catch (e) {
      Alert.alert('Error', 'Failed to sign in with Google');
    }
  };

  const handleGuest = async () => {
    try {
      await AsyncStorage.setItem('userToken', JSON.stringify({ guest: true }));
      navigation.reset({ index: 0, routes: [{ name: 'Dashboard' }] });
    } catch (e) {
      Alert.alert('Error', 'Failed to continue as guest');
    }
  };

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.container}>
      <View style={styles.box}>
        <Text style={styles.title}>Welcome to RevoScope</Text>
        <Text style={styles.subtitle}>{isSignUp ? 'Create your account' : 'Sign in to continue'}</Text>

        <TextInput
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
          style={styles.input}
        />

        <TextInput
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          style={styles.input}
        />

        {isSignUp && (
          <TextInput
            placeholder="Confirm Password"
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            secureTextEntry
            style={styles.input}
          />
        )}

        <TouchableOpacity 
          style={styles.button} 
          onPress={isSignUp ? handleSignUp : handleLogin}
        >
          <Text style={styles.buttonText}>{isSignUp ? 'Create Account' : 'Sign In'}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.dividerContainer}>
          <View style={styles.divider} />
          <Text style={styles.dividerText}>or</Text>
          <View style={styles.divider} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.googleButton} onPress={handleGoogleSignIn}>
          <Ionicons name="logo-google" size={20} color="#DB4437" />
          <Text style={styles.googleButtonText}>Sign in with Google</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.link} 
          onPress={() => {
            setIsSignUp(!isSignUp);
            setEmail('');
            setPassword('');
            setConfirmPassword('');
          }}
        >
          <Text style={styles.linkText}>
            {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.link} onPress={handleGuest}>
          <Text style={styles.linkText}>Continue as Guest</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F8FAFC' },
  box: { width: '90%', padding: 24, backgroundColor: '#fff', borderRadius: 12, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 10 },
  title: { fontSize: 20, fontWeight: '700', marginBottom: 4 },
  subtitle: { fontSize: 14, color: '#6B7280', marginBottom: 16 },
  input: { height: 48, borderColor: '#E5E7EB', borderWidth: 1, borderRadius: 8, paddingHorizontal: 12, marginBottom: 12, backgroundColor: '#FAFAFA' },
  button: { height: 48, backgroundColor: '#0EA5A4', borderRadius: 8, justifyContent: 'center', alignItems: 'center', marginTop: 8 },
  buttonText: { color: '#fff', fontWeight: '600' },
  dividerContainer: { flexDirection: 'row', alignItems: 'center', marginVertical: 16 },
  divider: { flex: 1, height: 1, backgroundColor: '#E5E7EB' },
  dividerText: { marginHorizontal: 8, color: '#9CA3AF', fontSize: 12 },
  googleButton: { height: 48, borderColor: '#E5E7EB', borderWidth: 1, borderRadius: 8, justifyContent: 'center', alignItems: 'center', flexDirection: 'row', backgroundColor: '#FAFAFA' },
  googleButtonText: { color: '#1F2937', fontWeight: '600', marginLeft: 8 },
  link: { marginTop: 12, alignItems: 'center' },
  linkText: { color: '#2563EB', fontSize: 14 }
});
