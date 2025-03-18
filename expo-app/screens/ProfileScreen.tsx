import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ScrollView, Image, Alert } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { supabase } from '../lib/supabase';
import { Tables } from '../lib/supabase';

type UserProfile = Tables['users']['Row'];
type VideoItem = Tables['videos']['Row'];

export default function ProfileScreen({ navigation }: any) {
  const { user, signOut } = useAuth();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchProfile();
      fetchVideos();
    }
  }, [user]);

  const fetchProfile = async () => {
    try {
      const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', user?.id)
        .single();

      if (error) throw error;
      setProfile(data);
    } catch (error: any) {
      console.error('Error fetching profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchVideos = async () => {
    try {
      const { data, error } = await supabase
        .from('videos')
        .select('*')
        .eq('user_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setVideos(data || []);
    } catch (error: any) {
      console.error('Error fetching videos:', error);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (error) {
      console.error('Error signing out:', error);
      Alert.alert('Error', 'Failed to sign out');
    }
  };

  const navigateToVideoDetail = (videoId: string) => {
    // TODO: Navigate to video detail screen
    console.log('Navigate to video:', videoId);
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <Text>Loading profile...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.profileHeader}>
        <Image 
          source={{ 
            uri: profile?.avatar_url || 'https://www.gravatar.com/avatar/?d=mp' 
          }} 
          style={styles.avatar} 
        />
        <View style={styles.profileInfo}>
          <Text style={styles.name}>{profile?.first_name} {profile?.last_name}</Text>
          <Text style={styles.email}>{user?.email}</Text>
        </View>
      </View>

      <TouchableOpacity 
        style={styles.editProfileButton} 
        onPress={() => {
          // TODO: Navigate to edit profile screen
          console.log('Navigate to edit profile');
        }}
      >
        <Text style={styles.editProfileText}>Edit Profile</Text>
      </TouchableOpacity>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Your Videos</Text>
        
        {videos.length === 0 ? (
          <Text style={styles.emptyState}>You haven't uploaded any videos yet.</Text>
        ) : (
          videos.map((video) => (
            <TouchableOpacity
              key={video.id}
              style={styles.videoItem}
              onPress={() => navigateToVideoDetail(video.id)}
            >
              <View style={styles.videoThumbnail}>
                {video.thumbnail_url ? (
                  <Image 
                    source={{ uri: video.thumbnail_url }} 
                    style={styles.thumbnailImage} 
                  />
                ) : (
                  <View style={styles.thumbnailPlaceholder}>
                    <Text>No Thumbnail</Text>
                  </View>
                )}
              </View>
              <View style={styles.videoInfo}>
                <Text style={styles.videoTitle}>{video.title}</Text>
                <Text style={styles.videoDate}>
                  {new Date(video.created_at).toLocaleDateString()}
                </Text>
                <Text 
                  style={[
                    styles.videoStatus, 
                    video.analysis_status === 'completed' ? styles.statusCompleted :
                    video.analysis_status === 'failed' ? styles.statusFailed :
                    styles.statusPending
                  ]}
                >
                  {video.analysis_status.charAt(0).toUpperCase() + video.analysis_status.slice(1)}
                </Text>
              </View>
            </TouchableOpacity>
          ))
        )}
      </View>

      <TouchableOpacity style={styles.signOutButton} onPress={handleSignOut}>
        <Text style={styles.signOutText}>Sign Out</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#eee',
  },
  profileInfo: {
    marginLeft: 20,
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  email: {
    fontSize: 16,
    color: '#666',
    marginTop: 4,
  },
  editProfileButton: {
    margin: 20,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
  },
  editProfileText: {
    color: '#0077B6',
    fontWeight: 'bold',
  },
  section: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  emptyState: {
    textAlign: 'center',
    color: '#888',
    fontSize: 16,
    marginTop: 20,
    marginBottom: 20,
  },
  videoItem: {
    flexDirection: 'row',
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#eee',
    borderRadius: 8,
    overflow: 'hidden',
  },
  videoThumbnail: {
    width: 120,
    height: 80,
  },
  thumbnailImage: {
    width: '100%',
    height: '100%',
  },
  thumbnailPlaceholder: {
    width: '100%',
    height: '100%',
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  videoInfo: {
    flex: 1,
    padding: 10,
    justifyContent: 'center',
  },
  videoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  videoDate: {
    fontSize: 14,
    color: '#888',
    marginBottom: 4,
  },
  videoStatus: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  statusCompleted: {
    color: '#4CAF50',
  },
  statusPending: {
    color: '#FFC107',
  },
  statusFailed: {
    color: '#F44336',
  },
  signOutButton: {
    margin: 20,
    padding: 15,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
  },
  signOutText: {
    color: '#F44336',
    fontWeight: 'bold',
  },
});