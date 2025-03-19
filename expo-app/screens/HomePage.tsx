import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  Image,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../types';

type Props = {
  username?: string;
};

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

const HomePage: React.FC<Props> = ({ username = 'Player' }) => {
  const navigation = useNavigation<HomeScreenNavigationProp>();
  const [activeTab, setActiveTab] = useState<'feed' | 'stats' | 'coach'>('feed');

  const renderTabContent = () => {
    switch (activeTab) {
      case 'feed':
        return <FeedTab navigation={navigation} />;
      case 'stats':
        return <StatsTab />;
      case 'coach':
        return <CoachTab />;
      default:
        return <FeedTab navigation={navigation} />;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.greeting}>Hello, {username}</Text>
        <TouchableOpacity>
          <Ionicons name="notifications-outline" size={24} color="#333" />
        </TouchableOpacity>
      </View>

      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'feed' && styles.activeTab]}
          onPress={() => setActiveTab('feed')}
        >
          <Text style={[styles.tabText, activeTab === 'feed' && styles.activeTabText]}>Feed</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'stats' && styles.activeTab]}
          onPress={() => setActiveTab('stats')}
        >
          <Text style={[styles.tabText, activeTab === 'stats' && styles.activeTabText]}>My Stats</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'coach' && styles.activeTab]}
          onPress={() => setActiveTab('coach')}
        >
          <Text style={[styles.tabText, activeTab === 'coach' && styles.activeTabText]}>Coach</Text>
        </TouchableOpacity>
      </View>

      {renderTabContent()}

      <View style={styles.actionButton}>
        <TouchableOpacity
          style={styles.recordButton}
          onPress={() => navigation.navigate('VideoUploader')}
        >
          <Ionicons name="videocam" size={30} color="#fff" />
          <Text style={styles.recordButtonText}>Record Swing</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

type FeedTabProps = {
  navigation: HomeScreenNavigationProp;
};

const FeedTab: React.FC<FeedTabProps> = ({ navigation }) => {
  return (
    <ScrollView style={styles.tabContent}>
      <TouchableOpacity style={styles.card} onPress={() => navigation.navigate('VideoAnalysis', { videoId: 'demo-1', taskId: 'demo-task-1' })}>
        <Image
          source={{ uri: 'https://images.unsplash.com/photo-1594381898411-846e7d193883?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTh8fHRlbm5pc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60' }}
          style={styles.cardImage}
          resizeMode="cover"
        />
        <View style={styles.cardContent}>
          <Text style={styles.cardTitle}>Forehand Analysis</Text>
          <Text style={styles.cardDate}>June 15, 2023</Text>
          <View style={styles.cardStats}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>85</Text>
              <Text style={styles.statLabel}>Score</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>72mph</Text>
              <Text style={styles.statLabel}>Speed</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>Good</Text>
              <Text style={styles.statLabel}>Form</Text>
            </View>
          </View>
        </View>
      </TouchableOpacity>

      <TouchableOpacity style={styles.card} onPress={() => navigation.navigate('VideoAnalysis', { videoId: 'demo-2', taskId: 'demo-task-2' })}>
        <Image
          source={{ uri: 'https://images.unsplash.com/photo-1622279457486-62dcc4a431d6?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NHx8dGVubmlzJTIwcGxheWVyfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60' }}
          style={styles.cardImage}
          resizeMode="cover"
        />
        <View style={styles.cardContent}>
          <Text style={styles.cardTitle}>Backhand Practice</Text>
          <Text style={styles.cardDate}>June 10, 2023</Text>
          <View style={styles.cardStats}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>78</Text>
              <Text style={styles.statLabel}>Score</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>68mph</Text>
              <Text style={styles.statLabel}>Speed</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>Fair</Text>
              <Text style={styles.statLabel}>Form</Text>
            </View>
          </View>
        </View>
      </TouchableOpacity>
    </ScrollView>
  );
};

const StatsTab: React.FC = () => {
  return (
    <ScrollView style={styles.tabContent}>
      <View style={styles.statsContainer}>
        <Text style={styles.statsTitle}>Your Progress</Text>
        <View style={styles.statsChart}>
          <Text style={styles.chartPlaceholder}>Progress chart will appear here</Text>
        </View>
        
        <Text style={styles.statsTitle}>Key Metrics</Text>
        <View style={styles.metricsContainer}>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>75</Text>
            <Text style={styles.metricLabel}>Avg. Score</Text>
          </View>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>68 mph</Text>
            <Text style={styles.metricLabel}>Avg. Speed</Text>
          </View>
          <View style={styles.metricCard}>
            <Text style={styles.metricValue}>12</Text>
            <Text style={styles.metricLabel}>Sessions</Text>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const CoachTab: React.FC = () => {
  return (
    <ScrollView style={styles.tabContent}>
      <View style={styles.coachContainer}>
        <Text style={styles.coachTitle}>Coach's Tips</Text>
        <View style={styles.tipCard}>
          <Ionicons name="bulb-outline" size={24} color="#FF9500" style={styles.tipIcon} />
          <View style={styles.tipContent}>
            <Text style={styles.tipTitle}>Improve Your Forehand</Text>
            <Text style={styles.tipDescription}>
              Work on hip rotation during your swing to generate more power and maintain better balance.
            </Text>
          </View>
        </View>
        <View style={styles.tipCard}>
          <Ionicons name="fitness-outline" size={24} color="#FF9500" style={styles.tipIcon} />
          <View style={styles.tipContent}>
            <Text style={styles.tipTitle}>Footwork Drill</Text>
            <Text style={styles.tipDescription}>
              Try the ladder drill to improve your footwork and court coverage. This will help you reach more shots.
            </Text>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f7',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  greeting: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  tabContainer: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#e1e1e1',
    backgroundColor: '#fff',
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#2196F3',
  },
  tabText: {
    fontSize: 14,
    color: '#888',
  },
  activeTabText: {
    color: '#2196F3',
    fontWeight: 'bold',
  },
  tabContent: {
    flex: 1,
  },
  actionButton: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  recordButton: {
    flexDirection: 'row',
    backgroundColor: '#2196F3',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 30,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    alignItems: 'center',
  },
  recordButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginHorizontal: 16,
    marginTop: 16,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  cardImage: {
    width: '100%',
    height: 180,
  },
  cardContent: {
    padding: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  cardDate: {
    fontSize: 14,
    color: '#888',
    marginBottom: 12,
  },
  cardStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  statLabel: {
    fontSize: 12,
    color: '#888',
  },
  statsContainer: {
    padding: 16,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    marginTop: 8,
  },
  statsChart: {
    backgroundColor: '#fff',
    height: 200,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
    padding: 16,
  },
  chartPlaceholder: {
    color: '#888',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metricCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    width: '30%',
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2196F3',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 12,
    color: '#888',
  },
  coachContainer: {
    padding: 16,
  },
  coachTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  tipCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    flexDirection: 'row',
  },
  tipIcon: {
    marginRight: 16,
  },
  tipContent: {
    flex: 1,
  },
  tipTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  tipDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});

export default HomePage;