import React, { useState } from 'react';
import { View, Text, StyleSheet, StatusBar, SafeAreaView, TouchableOpacity } from 'react-native';
import { Button } from '../components/ui/Button';
import { ChevronLeft } from '../components/ui/Icons';

// Comment out imports for components that will be implemented later
// import TrendsView from './TrendsView';
// import MatchView from './MatchView';
// import ShotPlacementView from './ShotPlacementView';

// Create placeholder components
const TrendsView = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
    <Text style={{ color: 'white' }}>Trends View (Coming Soon)</Text>
  </View>
);

const MatchView = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
    <Text style={{ color: 'white' }}>Match View (Coming Soon)</Text>
  </View>
);

const ShotPlacementView = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
    <Text style={{ color: 'white' }}>Shot Placement View (Coming Soon)</Text>
  </View>
);

type TennisAnalyticsDashboardProps = {
  onBack?: () => void;
};

export default function TennisAnalyticsDashboard({ onBack }: TennisAnalyticsDashboardProps) {
  const [currentView, setCurrentView] = useState<'trends' | 'match' | 'shot-placement'>('trends');

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.content}>
        {currentView === 'trends' && (
          <>
            <View style={styles.header}>
              {onBack && (
                <Button
                  variant="ghost"
                  size="icon"
                  style={styles.backButton}
                  onPress={onBack}
                >
                  <ChevronLeft size={20} color="#a5b4fc" />
                </Button>
              )}
              <Text style={styles.title}>Performance Trends</Text>
            </View>
            <TrendsView />
          </>
        )}

        {currentView === 'match' && (
          <>
            <View style={styles.header}>
              <TouchableOpacity
                style={styles.backLinkButton}
                onPress={() => setCurrentView('trends')}
              >
                <ChevronLeft size={20} color="#a5b4fc" />
                <Text style={styles.backLinkText}>Back</Text>
              </TouchableOpacity>
              <Text style={styles.title}>Match Analysis</Text>
            </View>
            <MatchView />
          </>
        )}

        {currentView === 'shot-placement' && (
          <>
            <View style={styles.header}>
              <TouchableOpacity
                style={styles.backLinkButton}
                onPress={() => setCurrentView('match')}
              >
                <ChevronLeft size={20} color="#a5b4fc" />
                <Text style={styles.backLinkText}>Back</Text>
              </TouchableOpacity>
              <Text style={styles.title}>Shot Distribution</Text>
            </View>
            <ShotPlacementView />
          </>
        )}
      </View>

      {/* Navigation */}
      <View style={styles.navigation}>
        <TouchableOpacity
          style={[
            styles.navButton,
            currentView === 'trends' && styles.activeNavButton,
          ]}
          onPress={() => setCurrentView('trends')}
        >
          <Text
            style={[
              styles.navButtonText,
              currentView === 'trends' && styles.activeNavButtonText,
            ]}
          >
            Trends
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.navButton,
            currentView === 'match' && styles.activeNavButton,
          ]}
          onPress={() => setCurrentView('match')}
        >
          <Text
            style={[
              styles.navButtonText,
              currentView === 'match' && styles.activeNavButtonText,
            ]}
          >
            Match
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.navButton,
            currentView === 'shot-placement' && styles.activeNavButton,
          ]}
          onPress={() => setCurrentView('shot-placement')}
        >
          <Text
            style={[
              styles.navButtonText,
              currentView === 'shot-placement' && styles.activeNavButtonText,
            ]}
          >
            Shot Data
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a', // slate-950
  },
  content: {
    flex: 1,
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  backLinkButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16,
  },
  backLinkText: {
    fontSize: 16,
    color: '#a5b4fc', // indigo-400
    marginLeft: 4,
  },
  navigation: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#1e293b', // slate-800
    backgroundColor: 'rgba(15, 23, 42, 0.9)', // slate-900/80
  },
  navButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 12,
  },
  activeNavButton: {
    backgroundColor: 'rgba(79, 70, 229, 0.2)', // indigo-900/50
  },
  navButtonText: {
    fontSize: 16,
    color: '#94a3b8', // slate-400
  },
  activeNavButtonText: {
    color: '#a5b4fc', // indigo-300
  },
}); 