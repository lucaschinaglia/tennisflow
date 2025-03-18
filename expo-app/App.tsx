import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';
import { supabase } from './lib/supabase';
import { AuthProvider, useAuth } from './hooks/useAuth';

// Screens
import AuthScreen from './screens/AuthScreen';
import HomePage from './screens/HomePage';
import ProfileScreen from './screens/ProfileScreen';
import VideoUploader from './components/VideoUploader';

// Create navigators
const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

// Content when user is authenticated
function AuthenticatedApp() {
  const { user } = useAuth();
  
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;
          
          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'Upload') {
            iconName = focused ? 'cloud-upload' : 'cloud-upload-outline';
          } else if (route.name === 'Profile') {
            iconName = focused ? 'person' : 'person-outline';
          }
          
          return <Ionicons name={iconName as any} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#0077B6',
        tabBarInactiveTintColor: 'gray',
      })}
    >
      <Tab.Screen name="Home" component={HomePage} />
      <Tab.Screen 
        name="Upload" 
        component={VideoUploadScreen} 
        options={{ 
          title: 'Upload Video'
        }} 
      />
      <Tab.Screen 
        name="Profile" 
        component={ProfileScreen} 
        options={{
          title: 'My Profile'
        }}
      />
    </Tab.Navigator>
  );
}

// Simple wrapper for the VideoUploader component
function VideoUploadScreen() {
  return (
    <View style={styles.container}>
      <VideoUploader />
    </View>
  );
}

// Root component that handles authentication state
function RootNavigator() {
  const { user, loading } = useAuth();
  
  // Show loading screen while auth state is being determined
  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0077B6" />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }
  
  return (
    <NavigationContainer>
      {user ? (
        <AuthenticatedApp />
      ) : (
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          <Stack.Screen name="Auth" component={AuthScreen} />
        </Stack.Navigator>
      )}
    </NavigationContainer>
  );
}

// Main app component
export default function App() {
  const [isSupabaseReady, setIsSupabaseReady] = useState(false);
  
  useEffect(() => {
    // Check Supabase connection
    const checkConnection = async () => {
      try {
        const { data, error } = await supabase.from('users').select('count').single();
        if (error) {
          console.error('Supabase connection error:', error);
        }
        setIsSupabaseReady(true);
      } catch (error) {
        console.error('Failed to connect to Supabase:', error);
        setIsSupabaseReady(true); // Set to true anyway to not block the app
      }
    };
    
    checkConnection();
  }, []);
  
  if (!isSupabaseReady) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0077B6" />
        <Text style={styles.loadingText}>Connecting to database...</Text>
      </View>
    );
  }
  
  return (
    <AuthProvider>
      <StatusBar style="auto" />
      <RootNavigator />
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
});