import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Alert,
  Image,
  Animated,
  ScrollView,
  Dimensions,
  ActivityIndicator,
  Keyboard,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { loginUser, registerUser } from '../services/DatabaseService';

const { width } = Dimensions.get('window');

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [hospital, setHospital] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const animateFormSwitch = () => {
    Animated.sequence([
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const handleLogin = async () => {
    Keyboard.dismiss();
    if (!email.trim() || !password) {
      Alert.alert('Missing Details', 'Please enter your email and password.');
      return;
    }

    setLoading(true);
    try {
      const user = await loginUser(email, password);

      if (user) {
        await AsyncStorage.setItem('userToken', JSON.stringify({ ...user, online: true }));
        navigation.reset({ index: 0, routes: [{ name: 'Dashboard' }] });
      } else {
        Alert.alert('Login Failed', 'Invalid email or password.');
      }
    } catch (e) {
      console.error(e);
      Alert.alert('Error', 'An error occurred during login. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async () => {
    Keyboard.dismiss();
    if (!fullName.trim()) {
      Alert.alert('Missing Details', 'Please enter your full name.');
      return;
    }

    if (!email.trim() || !password) {
      Alert.alert('Missing Details', 'Please enter an email and password.');
      return;
    }

    if (!hospital.trim()) {
      Alert.alert('Missing Details', 'Please enter your hospital or clinic name.');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Password Mismatch', 'Passwords do not match.');
      return;
    }

    if (password.length < 6) {
      Alert.alert('Weak Password', 'Password must be at least 6 characters long.');
      return;
    }

    setLoading(true);
    try {
      // Register user in local DB
      await registerUser(email, password, fullName, hospital);

      // Auto login after registration
      const user = await loginUser(email, password);

      if (user) {
        await AsyncStorage.setItem(
          'userToken',
          JSON.stringify({ ...user, online: true })
        );
        navigation.reset({ index: 0, routes: [{ name: 'Dashboard' }] });
      } else {
        // Fallback if auto-login fails for some reason
        Alert.alert('Account Created', 'Please sign in with your new account.');
        toggleForm();
      }
    } catch (e) {
      console.error(e);
      if (e.message.includes('Email already registered')) {
        Alert.alert('Registration Failed', 'This email is already registered.');
      } else {
        Alert.alert('Error', 'Failed to create account. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    Alert.alert('Coming Soon', 'Google Sign-In will be implemented soon.');
  };

  const toggleForm = () => {
    animateFormSwitch();
    setIsSignUp(!isSignUp);
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setFullName('');
    setHospital('');
    // Clear any previous focus or keyboard state if needed
    Keyboard.dismiss();
  };

  const renderInput = (placeholder, value, onChangeText, options = {}) => (
    <View style={styles.inputContainer}>
      <Ionicons
        name={options.icon || 'mail-outline'}
        size={20}
        color="#9CA3AF"
        style={styles.inputIcon}
      />
      <TextInput
        placeholder={placeholder}
        placeholderTextColor="#9CA3AF"
        value={value}
        onChangeText={onChangeText}
        keyboardType={options.keyboardType || 'default'}
        autoCapitalize={options.autoCapitalize || 'none'}
        secureTextEntry={options.secureTextEntry && !showPassword}
        autoCorrect={false}
        spellCheck={false}
        textContentType={options.textContentType || 'none'}
        autoComplete={options.autoComplete || 'off'}
        style={[styles.input, { backgroundColor: 'transparent' }]}
        editable={!loading}
      />
      {options.secureTextEntry && (
        <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
          <Ionicons
            name={showPassword ? 'eye-outline' : 'eye-off-outline'}
            size={20}
            color="#9CA3AF"
          />
        </TouchableOpacity>
      )}
    </View>
  );

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Back Button */}
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => {
            if (navigation.canGoBack()) {
              navigation.goBack();
            } else {
              navigation.reset({
                index: 0,
                routes: [{ name: 'Welcome' }],
              });
            }
          }}
          disabled={loading}
        >
          <Ionicons name="arrow-back" size={24} color="#E85656" />
        </TouchableOpacity>

        <Animated.View
          style={[
            styles.content,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.iconContainer}>
              <Ionicons
                name={isSignUp ? 'person-add' : 'person'}
                size={32}
                color="#E85656"
              />
            </View>
            {isSignUp ? (
              <Text style={styles.title}>Create Account</Text>
            ) : (
              <Text style={styles.textLogo}>
                Revo<Text style={{ color: '#E85656' }}>Scope</Text>
              </Text>
            )}
            <Text style={styles.subtitle}>
              {isSignUp
                ? 'Sign up to sync your triage data securely'
                : 'Sign in to access your secure workspace'}
            </Text>
          </View>

          {/* Form */}
          <View style={styles.form}>
            {isSignUp && (
              <>
                {renderInput('Full Name', fullName, setFullName, {
                  icon: 'person-outline',
                  autoCapitalize: 'words',
                  textContentType: 'name',
                  autoComplete: 'name',
                })}
                {renderInput('Hospital / Clinic', hospital, setHospital, {
                  icon: 'business-outline',
                  autoCapitalize: 'words',
                  textContentType: 'organizationName',
                })}
              </>
            )}

            {renderInput('Email Address', email, setEmail, {
              icon: 'mail-outline',
              keyboardType: 'email-address',
              textContentType: 'emailAddress',
              autoComplete: 'email',
            })}

            {renderInput('Password', password, setPassword, {
              icon: 'lock-closed-outline',
              secureTextEntry: true,
              textContentType: 'password',
              autoComplete: 'password',
            })}

            {isSignUp &&
              renderInput('Confirm Password', confirmPassword, setConfirmPassword, {
                icon: 'lock-closed-outline',
                secureTextEntry: true,
                textContentType: 'password',
                autoComplete: 'password-new',
              })}

            {/* Submit Button */}
            <TouchableOpacity
              style={styles.submitButton}
              onPress={isSignUp ? handleSignUp : handleLogin}
              activeOpacity={0.85}
              disabled={loading}
            >
              <LinearGradient
                colors={['#E85656', '#D44848']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.gradientButton}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <>
                    <Text style={styles.submitButtonText}>
                      {isSignUp ? 'Create Account' : 'Sign In'}
                    </Text>
                    <Ionicons name="arrow-forward" size={20} color="#fff" />
                  </>
                )}
              </LinearGradient>
            </TouchableOpacity>

            {/* Divider */}
            <View style={styles.dividerContainer}>
              <View style={styles.divider} />
              <Text style={styles.dividerText}>or continue with</Text>
              <View style={styles.divider} />
            </View>

            {/* Google Sign In */}
            <TouchableOpacity
              style={styles.googleButton}
              onPress={handleGoogleSignIn}
              activeOpacity={0.85}
              disabled={loading}
            >
              <Image
                source={{
                  uri: 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/500px-Google_%22G%22_logo.svg.png',
                }}
                style={styles.googleIcon}
                resizeMode="contain"
              />
              <Text style={styles.googleButtonText}>Google</Text>
            </TouchableOpacity>
          </View>

          {/* Toggle Link */}
          <View style={styles.toggleContainer}>
            <Text style={styles.toggleText}>
              {isSignUp ? 'Already have an account?' : "Don't have an account?"}
            </Text>
            <TouchableOpacity onPress={toggleForm} disabled={loading}>
              <Text style={styles.toggleLink}>
                {isSignUp ? ' Sign In' : ' Sign Up'}
              </Text>
            </TouchableOpacity>
          </View>
        </Animated.View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFF8F8',
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
    marginBottom: 20,
  },
  content: {
    flex: 1,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  iconContainer: {
    width: 72,
    height: 72,
    borderRadius: 20,
    backgroundColor: '#FEE2E2',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#E85656',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 5,
  },
  title: {
    fontSize: 26,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 8,
    textAlign: 'center',
  },
  textLogo: {
    fontSize: 32,
    fontWeight: '800',
    color: '#1F2937',
    marginBottom: 8,
    textAlign: 'center',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 15,
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 22,
    maxWidth: '80%',
  },
  form: {
    width: '100%',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 16,
    paddingHorizontal: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#F3F4F6',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 6,
    elevation: 2,
    height: 56,
  },
  inputIcon: {
    marginRight: 12,
  },
  input: {
    flex: 1,
    height: '100%',
    fontSize: 16,
    color: '#1F2937',
  },
  submitButton: {
    marginTop: 12,
    borderRadius: 16,
    shadowColor: '#E85656',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 8,
  },
  gradientButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    borderRadius: 16,
    gap: 8,
  },
  submitButtonText: {
    fontSize: 17,
    fontWeight: '700',
    color: '#fff',
  },
  dividerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 24,
  },
  divider: {
    flex: 1,
    height: 1,
    backgroundColor: '#E5E7EB',
  },
  dividerText: {
    marginHorizontal: 16,
    color: '#9CA3AF',
    fontSize: 13,
    fontWeight: '500',
  },
  googleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    backgroundColor: '#fff',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    gap: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 6,
    elevation: 2,
  },
  googleIcon: {
    width: 22,
    height: 22,
  },
  googleButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  toggleContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 32,
    marginBottom: 20,
  },
  toggleText: {
    fontSize: 15,
    color: '#6B7280',
  },
  toggleLink: {
    fontSize: 15,
    fontWeight: '700',
    color: '#E85656',
  },
});
