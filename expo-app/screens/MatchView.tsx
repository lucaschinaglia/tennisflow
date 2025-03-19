import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Card } from '../components/ui/Card';

export default function MatchView() {
  return (
    <View style={styles.container}>
      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Recent Match</Text>
        <View style={styles.matchHeader}>
          <View style={styles.playerScore}>
            <Text style={styles.playerName}>You</Text>
            <Text style={styles.score}>6-4, 7-5</Text>
          </View>
          <Text style={styles.vs}>vs</Text>
          <View style={styles.playerScore}>
            <Text style={styles.playerName}>Opponent</Text>
            <Text style={styles.score}>4-6, 5-7</Text>
          </View>
        </View>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Match Statistics</Text>
        
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Aces</Text>
          <View style={styles.statBars}>
            <View style={[styles.statBar, { width: '40%' }]}>
              <Text style={styles.statBarText}>4</Text>
            </View>
            <View style={[styles.statBarOpponent, { width: '20%' }]}>
              <Text style={styles.statBarText}>2</Text>
            </View>
          </View>
        </View>

        <View style={styles.statItem}>
          <Text style={styles.statLabel}>First Serve %</Text>
          <View style={styles.statBars}>
            <View style={[styles.statBar, { width: '65%' }]}>
              <Text style={styles.statBarText}>65%</Text>
            </View>
            <View style={[styles.statBarOpponent, { width: '55%' }]}>
              <Text style={styles.statBarText}>55%</Text>
            </View>
          </View>
        </View>

        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Winners</Text>
          <View style={styles.statBars}>
            <View style={[styles.statBar, { width: '38%' }]}>
              <Text style={styles.statBarText}>19</Text>
            </View>
            <View style={[styles.statBarOpponent, { width: '24%' }]}>
              <Text style={styles.statBarText}>12</Text>
            </View>
          </View>
        </View>

        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Unforced Errors</Text>
          <View style={styles.statBars}>
            <View style={[styles.statBar, { width: '30%' }]}>
              <Text style={styles.statBarText}>15</Text>
            </View>
            <View style={[styles.statBarOpponent, { width: '44%' }]}>
              <Text style={styles.statBarText}>22</Text>
            </View>
          </View>
        </View>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Key Moments</Text>
        <View style={styles.keyMoment}>
          <Text style={styles.keyMomentTitle}>Break Point at 4-4 (Set 1)</Text>
          <Text style={styles.keyMomentDesc}>Strong forehand winner down the line</Text>
        </View>
        <View style={styles.keyMoment}>
          <Text style={styles.keyMomentTitle}>Set Point at 6-5 (Set 2)</Text>
          <Text style={styles.keyMomentDesc}>Ace down the T</Text>
        </View>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  card: {
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  matchHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#1e293b', // slate-800
    borderRadius: 8,
    padding: 16,
  },
  playerScore: {
    alignItems: 'center',
  },
  playerName: {
    color: '#e2e8f0', // slate-200
    fontSize: 16,
    marginBottom: 4,
  },
  score: {
    color: '#a5b4fc', // indigo-300
    fontSize: 18,
    fontWeight: 'bold',
  },
  vs: {
    color: '#64748b', // slate-500
    fontSize: 14,
  },
  statItem: {
    marginBottom: 12,
  },
  statLabel: {
    color: '#e2e8f0', // slate-200
    marginBottom: 4,
  },
  statBars: {
    flexDirection: 'row',
    height: 24,
    justifyContent: 'flex-start',
  },
  statBar: {
    backgroundColor: '#4f46e5', // indigo-600
    height: 24,
    borderRadius: 4,
    justifyContent: 'center',
    paddingHorizontal: 8,
    marginRight: 2,
  },
  statBarOpponent: {
    backgroundColor: '#1e40af', // blue-800
    height: 24,
    borderRadius: 4,
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  statBarText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
  },
  keyMoment: {
    backgroundColor: '#1e293b', // slate-800
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  keyMomentTitle: {
    color: '#a5b4fc', // indigo-300
    fontWeight: '600',
    marginBottom: 4,
  },
  keyMomentDesc: {
    color: '#94a3b8', // slate-400
    fontSize: 14,
  },
}); 