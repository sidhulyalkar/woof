import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Trophy, Medal, Award, Crown, Star, Zap, Target, Heart, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { EmojiAvatar, getEmojiForId, getVariantForId } from '@/components/ui/EmojiAvatar';

const mockLeaderboards = {
  weekly: [
    {
      rank: 1,
      pet: { name: 'Luna', type: 'Border Collie', avatar: 'https://images.unsplash.com/photo-1551717743-49959800b1f6?w=150&h=150&fit=crop&crop=face' },
      owner: 'Emma Davis',
      distance: '24.5 km',
      score: 2450
    },
    {
      rank: 2,
      pet: { name: 'Max', type: 'German Shepherd', avatar: 'https://images.unsplash.com/photo-1589941013453-ec89f33b5e95?w=150&h=150&fit=crop&crop=face' },
      owner: 'John Smith',
      distance: '22.1 km',
      score: 2210
    },
    {
      rank: 3,
      pet: { name: 'Buddy', type: 'Golden Retriever', avatar: 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=150&h=150&fit=crop&crop=face' },
      owner: 'Sarah Johnson',
      distance: '19.8 km',
      score: 1980
    },
    {
      rank: 4,
      pet: { name: 'Rocky', type: 'Labrador', avatar: 'https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=150&h=150&fit=crop&crop=face' },
      owner: 'Mike Wilson',
      distance: '17.3 km',
      score: 1730
    },
    {
      rank: 5,
      pet: { name: 'Bella', type: 'Husky', avatar: 'https://images.unsplash.com/photo-1605568427561-40dd23c2acea?w=150&h=150&fit=crop&crop=face' },
      owner: 'Lisa Chen',
      distance: '16.9 km',
      score: 1690
    }
  ],
  monthly: [
    {
      rank: 1,
      pet: { name: 'Thunder', type: 'Australian Shepherd', avatar: 'https://images.unsplash.com/photo-1680484539457-cd323f9e9ac5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhdXN0cmFsaWFuJTIwc2hlcGhlcmQlMjBkb2clMjBydW5uaW5nfGVufDF8fHx8MTc1OTE5OTE3Nnww&ixlib=rb-4.1.0&q=80&w=1080' },
      owner: 'David Miller',
      distance: '98.2 km',
      score: 9820
    },
    {
      rank: 2,
      pet: { name: 'Luna', type: 'Border Collie', avatar: 'https://images.unsplash.com/photo-1640958904594-fd2ed0b00167?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib3JkZXIlMjBjb2xsaWUlMjBhZ2lsaXR5JTIwY291cnNlJTIwanVtcGluZ3xlbnwxfHx8fDE3NTkxOTkwNjh8MA&ixlib=rb-4.1.0&q=80&w=1080' },
      owner: 'Emma Davis',
      distance: '94.7 km',
      score: 9470
    },
    {
      rank: 3,
      pet: { name: 'Shadow', type: 'Belgian Malinois', avatar: 'https://images.unsplash.com/photo-1605157478382-4604338a01ed?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxiZWxnaWFuJTIwbWFsaW5vaXMlMjBkb2clMjBhY3Rpb258ZW58MXx8fHwxNzU5MTk5MTc5fDA&ixlib=rb-4.1.0&q=80&w=1080' },
      owner: 'Alex Rodriguez',
      distance: '87.5 km',
      score: 8750
    }
  ]
};

const achievements = [
  { name: 'Marathon Master', icon: Trophy, description: '100km in a month', earned: true },
  { name: 'Daily Walker', icon: Medal, description: '7 days in a row', earned: true },
  { name: 'Speed Demon', icon: Award, description: 'Fastest 5km time', earned: false },
  { name: 'Social Butterfly', icon: Crown, description: '50 friends', earned: true }
];

export function LeaderboardScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('weekly');

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return (
          <div className="w-8 h-8 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full flex items-center justify-center">
            <Crown className="text-white" size={16} />
          </div>
        );
      case 2:
        return (
          <div className="w-8 h-8 bg-gradient-to-br from-gray-300 to-gray-500 rounded-full flex items-center justify-center">
            <Medal className="text-white" size={16} />
          </div>
        );
      case 3:
        return (
          <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-amber-700 rounded-full flex items-center justify-center">
            <Award className="text-white" size={16} />
          </div>
        );
      default:
        return (
          <div className="w-8 h-8 bg-muted/50 rounded-full flex items-center justify-center">
            <span className="text-sm font-semibold">{rank}</span>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50/30">
      {/* Header */}
      <div className="sticky top-0 z-50 backdrop-blur-2xl bg-white/90 border-b border-gray-200/40 px-6 py-4">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Leaderboards
          </h1>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-6 pb-24 space-y-6">
        {/* User's Current Rank */}
        <div className="bg-white/80 backdrop-blur-xl border border-gray-200/60 p-6 rounded-3xl shadow-sm">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center shadow-lg">
              <Trophy size={24} className="text-white" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-[17px] text-gray-900">Your Ranking</h3>
              <p className="text-sm text-gray-600">#8 this week • #12 this month</p>
            </div>
            <div className="text-right">
              <p className="text-xl font-bold text-blue-600">1,420 pts</p>
              <p className="text-sm text-gray-500">+180 today</p>
            </div>
          </div>
          <div className="mt-4">
            <div className="flex justify-between text-sm mb-2 text-gray-700">
              <span>To reach #7</span>
              <span className="font-medium">65 pts needed</span>
            </div>
            <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full" style={{ width: '85%' }}></div>
            </div>
          </div>
        </div>

        {/* Achievements */}
        <div className="glass-card p-6 rounded-xl">
          <div className="flex items-center gap-2 mb-4">
            <Star size={20} className="text-accent" />
            <h2 className="text-lg font-semibold">Recent Achievements</h2>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {achievements.map((achievement, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg transition-all duration-300 ${
                  achievement.earned
                    ? 'bg-accent/10 border border-accent/30 hover:bg-accent/20'
                    : 'bg-muted/20 border border-muted/30 opacity-60'
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    achievement.earned ? 'bg-accent/20' : 'bg-muted/30'
                  }`}>
                    <achievement.icon 
                      size={16} 
                      className={achievement.earned ? 'text-accent' : 'text-muted-foreground'} 
                    />
                  </div>
                  <span className="text-sm font-semibold">{achievement.name}</span>
                </div>
                <p className="text-xs text-muted-foreground">{achievement.description}</p>
                {achievement.earned && (
                  <div className="mt-2">
                    <Badge variant="secondary" className="text-xs bg-success/20 text-success border-success/30">
                      Unlocked!
                    </Badge>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Leaderboards */}
        <div className="glass-card p-6 rounded-xl">
          <div className="flex items-center gap-2 mb-4">
            <Trophy size={20} className="text-accent" />
            <h2 className="text-lg font-semibold">Top Performers</h2>
          </div>
          
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="weekly">This Week</TabsTrigger>
              <TabsTrigger value="monthly">This Month</TabsTrigger>
            </TabsList>
            
            <TabsContent value="weekly" className="space-y-4">
              {/* Top 3 Podium */}
              {mockLeaderboards.weekly.slice(0, 3).map((entry, index) => (
                <div
                  key={entry.rank}
                  className={`relative overflow-hidden rounded-3xl p-4 border cursor-pointer hover:shadow-lg transition-all duration-200 active:scale-[0.98] ${
                    entry.rank === 1 ? 'bg-gradient-to-r from-yellow-400/20 to-yellow-500/20 border-yellow-500/40' :
                    entry.rank === 2 ? 'bg-gradient-to-r from-gray-300/20 to-gray-400/20 border-gray-400/40' :
                    'bg-gradient-to-r from-amber-400/20 to-amber-500/20 border-amber-500/40'
                  }`}
                >
                  <div className="flex items-center gap-4">
                    {getRankIcon(entry.rank)}
                    <EmojiAvatar
                      emoji={getEmojiForId(entry.pet.name, 'pet')}
                      variant={getVariantForId(entry.pet.name)}
                      size="lg"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-[17px] text-gray-900">{entry.pet.name}</span>
                        <span className="text-gray-300">·</span>
                        <span className="text-sm font-medium text-gray-600">{entry.pet.type}</span>
                      </div>
                      <p className="text-sm text-gray-600">with {entry.owner}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xl font-bold text-blue-600">{entry.distance}</p>
                      <p className="text-sm text-gray-500">{entry.score} points</p>
                    </div>
                  </div>
                  {entry.rank === 1 && (
                    <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-yellow-400 to-yellow-500 rounded-full flex items-center justify-center shadow-lg border-2 border-white">
                      <Star size={14} className="text-white fill-white" />
                    </div>
                  )}
                </div>
              ))}

              {/* Rest of the leaderboard */}
              <div className="space-y-2 mt-4">
                {mockLeaderboards.weekly.slice(3).map((entry) => (
                  <div
                    key={entry.rank}
                    className="flex items-center gap-4 p-3 bg-surface-elevated/30 rounded-lg hover:bg-surface-elevated/50 transition-all duration-300"
                  >
                    {getRankIcon(entry.rank)}
                    <Avatar className="w-10 h-10">
                      <AvatarImage src={entry.pet.avatar} />
                      <AvatarFallback>{entry.pet.name[0]}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{entry.pet.name}</span>
                        <Badge variant="outline" className="text-xs">{entry.pet.type}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">with {entry.owner}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold">{entry.distance}</p>
                      <p className="text-sm text-muted-foreground">{entry.score} pts</p>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
            
            <TabsContent value="monthly" className="space-y-4">
              {mockLeaderboards.monthly.map((entry) => (
                <div
                  key={entry.rank}
                  className={`relative overflow-hidden rounded-xl p-4 ${
                    entry.rank === 1 ? 'bg-gradient-to-r from-yellow-500/20 to-yellow-600/20 border border-yellow-500/30' :
                    entry.rank === 2 ? 'bg-gradient-to-r from-gray-400/20 to-gray-500/20 border border-gray-400/30' :
                    entry.rank === 3 ? 'bg-gradient-to-r from-amber-500/20 to-amber-600/20 border border-amber-500/30' :
                    'bg-surface-elevated/30'
                  }`}
                >
                  <div className="flex items-center gap-4">
                    {getRankIcon(entry.rank)}
                    <Avatar className="w-14 h-14 ring-2 ring-accent/30">
                      <AvatarImage src={entry.pet.avatar} />
                      <AvatarFallback className="bg-accent/20">{entry.pet.name[0]}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-lg">{entry.pet.name}</span>
                        <Badge 
                          variant="secondary" 
                          className="text-xs bg-surface-elevated/50 border-accent/30"
                        >
                          {entry.pet.type}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">with {entry.owner}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xl font-bold text-accent">{entry.distance}</p>
                      <p className="text-sm text-muted-foreground">{entry.score} points</p>
                    </div>
                  </div>
                </div>
              ))}
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}