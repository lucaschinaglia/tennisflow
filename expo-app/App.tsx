import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useEffect, useState } from 'react';
import { View, Text } from 'react-native';
import { createClient } from '@supabase/supabase-js';
import 'react-native-url-polyfill/auto';
import HomePage from './screens/HomePage';

// Initialize Supabase
// Replace with your own Supabase URL and anon key
const supabaseUrl = 'YOUR_SUPABASE_URL';
const supabaseAnonKey = 'YOUR_SUPABASE_ANON_KEY';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export default function App() {
  const [isSupabaseConnected, setIsSupabaseConnected] = useState<boolean | null>(null);
  
  // Check Supabase connection on app start
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const { data, error } = await supabase.from('test_connection').select('*').limit(1);
        
        if (error) {
          console.error('Supabase connection error:', error);
          setIsSupabaseConnected(false);
          return;
        }
        
        setIsSupabaseConnected(true);
      } catch (err) {
        console.error('Error checking Supabase connection:', err);
        setIsSupabaseConnected(false);
      }
    };
    
    checkConnection();
  }, []);
  
  // Show connection status while checking
  if (isSupabaseConnected === null) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f172a' }}>
        <Text style={{ color: '#fff', fontSize: 16 }}>Connecting to database...</Text>
      </View>
    );
  }
  
  // Show error if connection fails
  if (isSupabaseConnected === false) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f172a' }}>
        <Text style={{ color: '#ef4444', fontSize: 16, marginBottom: 8 }}>Database connection failed</Text>
        <Text style={{ color: '#94a3b8', fontSize: 14, textAlign: 'center', paddingHorizontal: 20 }}>
          Check your internet connection or make sure Supabase credentials are correctly configured.
        </Text>
      </View>
    );
  }
  
  // If connected, show the app
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <HomePage />
    </SafeAreaProvider>
  );
}