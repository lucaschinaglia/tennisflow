import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { signIn, signUp, resetPassword } from '../services/supabaseService';
import type { RootStackParamList } from '../types';

type AuthScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Auth'>;

type AuthMode = 'login' | 'signup' | 'reset';

const AuthScreen: React.FC = () => {
  const navigation = useNavigation<AuthScreenNavigationProp>();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [authMode, setAuthMode] = useState<AuthMode>('login');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleAuth = async () => {
    if (!email.trim()) {
      Alert.alert('Error', 'Please enter your email');
      return;
    }

    if (authMode !== 'reset' && !password.trim()) {
      Alert.alert('Error', 'Please enter your password');
      return;
    }

    if (authMode === 'signup' && password !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }

    try {
      setLoading(true);

      if (authMode === 'login') {
        const { error } = await signIn(email, password);
        if (error) throw error;
        navigation.navigate('Home');
      } else if (authMode === 'signup') {
        const { error } = await signUp(email, password);
        if (error) throw error;
        Alert.alert(
          'Verification Email Sent',
          'Please check your email to verify your account.'
        );
        setAuthMode('login');
      } else if (authMode === 'reset') {
        const { error } = await resetPassword(email);
        if (error) throw error;
        Alert.alert(
          'Reset Email Sent',
          'Please check your email for password reset instructions.'
        );
        setAuthMode('login');
      }
    } catch (error: any) {
      Alert.alert('Error', error.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getTitle = () => {
    switch (authMode) {
      case 'login':
        return 'Welcome Back';
      case 'signup':
        return 'Create Account';
      case 'reset':
        return 'Reset Password';
      default:
        return 'Tennis Flow';
    }
  };

  const getSubtitle = () => {
    switch (authMode) {
      case 'login':
        return 'Sign in to continue';
      case 'signup':
        return 'Create a new account';
      case 'reset':
        return "We'll send you a reset link";
      default:
        return '';
    }
  };

  const getButtonText = () => {
    switch (authMode) {
      case 'login':
        return 'Sign In';
      case 'signup':
        return 'Sign Up';
      case 'reset':
        return 'Send Reset Link';
      default:
        return 'Continue';
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <StatusBar style="dark" />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.logoContainer}>
          <Image
            source={{ uri: 'https://via.placeholder.com/150?text=TF' }}
            style={styles.logo}
          />
          <Text style={styles.appName}>TennisFlow</Text>
        </View>

        <View style={styles.formContainer}>
          <Text style={styles.title}>{getTitle()}</Text>
          <Text style={styles.subtitle}>{getSubtitle()}</Text>

          <View style={styles.inputContainer}>
            <Ionicons name="mail-outline" size={20} color="#666" style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Email"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
              autoComplete="email"
              returnKeyType="next"
            />
          </View>

          {authMode !== 'reset' && (
            <View style={styles.inputContainer}>
              <Ionicons name="lock-closed-outline" size={20} color="#666" style={styles.inputIcon} />
              <TextInput
                style={[styles.input, { flex: 1 }]}
                placeholder="Password"
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                returnKeyType={authMode === 'signup' ? 'next' : 'done'}
              />
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
                <Ionicons
                  name={showPassword ? 'eye-off-outline' : 'eye-outline'}
                  size={20}
                  color="#666"
                />
              </TouchableOpacity>
            </View>
          )}

          {authMode === 'signup' && (
            <View style={styles.inputContainer}>
              <Ionicons name="lock-closed-outline" size={20} color="#666" style={styles.inputIcon} />
              <TextInput
                style={[styles.input, { flex: 1 }]}
                placeholder="Confirm Password"
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry={!showPassword}
                returnKeyType="done"
              />
            </View>
          )}

          {authMode === 'login' && (
            <TouchableOpacity
              style={styles.forgotPassword}
              onPress={() => setAuthMode('reset')}
            >
              <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={styles.authButton}
            onPress={handleAuth}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator size="small" color="#FFF" />
            ) : (
              <Text style={styles.authButtonText}>{getButtonText()}</Text>
            )}
          </TouchableOpacity>

          <View style={styles.switchModeContainer}>
            {authMode === 'login' ? (
              <Text style={styles.switchModeText}>
                Don't have an account?{' '}
                <Text
                  style={styles.switchModeLink}
                  onPress={() => setAuthMode('signup')}
                >
                  Sign Up
                </Text>
              </Text>
            ) : (
              <Text style={styles.switchModeText}>
                Already have an account?{' '}
                <Text
                  style={styles.switchModeLink}
                  onPress={() => setAuthMode('login')}
                >
                  Sign In
                </Text>
              </Text>
            )}
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f7',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: 40,
  },
  logo: {
    width: 100,
    height: 100,
    borderRadius: 20,
  },
  appName: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 10,
    color: '#333',
  },
  formContainer: {
    width: '100%',
    maxWidth: 350,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 30,
    textAlign: 'center',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    borderRadius: 10,
    marginBottom: 16,
    paddingHorizontal: 16,
    height: 56,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  inputIcon: {
    marginRight: 12,
  },
  input: {
    flex: 1,
    height: '100%',
    fontSize: 16,
    color: '#333',
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginBottom: 24,
  },
  forgotPasswordText: {
    color: '#2196F3',
    fontSize: 14,
  },
  authButton: {
    backgroundColor: '#2196F3',
    borderRadius: 10,
    height: 56,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#2196F3',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  authButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  switchModeContainer: {
    marginTop: 24,
    alignItems: 'center',
  },
  switchModeText: {
    color: '#666',
    fontSize: 14,
  },
  switchModeLink: {
    color: '#2196F3',
    fontWeight: 'bold',
  },
});

export default AuthScreen;