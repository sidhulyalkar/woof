'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Calendar, MapPin, Clock, Send, Star, ThumbsUp, ThumbsDown, MessageSquare } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';
import { Avatar } from '@/components/ui/avatar';

interface MeetupProposal {
  id: string;
  proposerId: string;
  targetUserId: string;
  status: 'pending' | 'accepted' | 'rejected' | 'confirmed' | 'completed';
  proposedTime: string;
  locationName: string;
  lat?: number;
  lng?: number;
  activityType: string;
  message?: string;
  proposer: {
    id: string;
    handle: string;
    avatarUrl?: string;
  };
  target: {
    id: string;
    handle: string;
    avatarUrl?: string;
  };
  feedback?: {
    didMeetupHappen: boolean;
    rating?: number;
    feedbackTags?: string[];
    comments?: string;
  };
  createdAt: string;
}

interface MeetupProposalScreenProps {
  matchUserId?: string;
  onClose?: () => void;
}

export function MeetupProposalScreen({ matchUserId, onClose }: MeetupProposalScreenProps) {
  const [showCreateForm, setShowCreateForm] = useState(!!matchUserId);
  const [selectedProposal, setSelectedProposal] = useState<MeetupProposal | null>(null);
  const queryClient = useQueryClient();

  const { data: proposals, isLoading } = useQuery<MeetupProposal[]>({
    queryKey: ['meetup-proposals'],
    queryFn: () => apiClient.get<MeetupProposal[]>('/meetup-proposals'),
  });

  const pendingProposals = proposals?.filter(p => p.status === 'pending') || [];
  const upcomingMeetups = proposals?.filter(p => p.status === 'confirmed' && new Date(p.proposedTime) > new Date()) || [];
  const pastMeetups = proposals?.filter(p => p.status === 'completed') || [];

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Meetup Proposals</h1>
            <p className="text-sm text-muted-foreground">Plan IRL meetups with your matches</p>
          </div>
          {!showCreateForm && (
            <Button onClick={() => setShowCreateForm(true)}>
              <Send className="w-4 h-4 mr-2" />
              Propose Meetup
            </Button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Create Form */}
          {showCreateForm && (
            <CreateProposalForm
              targetUserId={matchUserId}
              onCancel={() => {
                setShowCreateForm(false);
                onClose?.();
              }}
              onSuccess={() => {
                setShowCreateForm(false);
                onClose?.();
              }}
            />
          )}

          {/* Pending Proposals */}
          {pendingProposals.length > 0 && (
            <div>
              <h3 className="text-lg font-bold mb-3">Pending Proposals ({pendingProposals.length})</h3>
              <div className="space-y-3">
                {pendingProposals.map(proposal => (
                  <ProposalCard
                    key={proposal.id}
                    proposal={proposal}
                    onSelect={() => setSelectedProposal(proposal)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Upcoming Meetups */}
          {upcomingMeetups.length > 0 && (
            <div>
              <h3 className="text-lg font-bold mb-3">Confirmed Meetups ({upcomingMeetups.length})</h3>
              <div className="space-y-3">
                {upcomingMeetups.map(proposal => (
                  <ProposalCard
                    key={proposal.id}
                    proposal={proposal}
                    onSelect={() => setSelectedProposal(proposal)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Past Meetups */}
          {pastMeetups.length > 0 && (
            <div>
              <h3 className="text-lg font-bold mb-3">Past Meetups</h3>
              <div className="space-y-3">
                {pastMeetups.map(proposal => (
                  <ProposalCard
                    key={proposal.id}
                    proposal={proposal}
                    isPast
                    onSelect={() => setSelectedProposal(proposal)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!showCreateForm && proposals && proposals.length === 0 && (
            <Card className="p-8 text-center">
              <Calendar className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">No meetup proposals yet</h3>
              <p className="text-muted-foreground mb-4">
                Start connecting with matches and propose your first meetup!
              </p>
              <Button onClick={() => setShowCreateForm(true)}>
                Propose a Meetup
              </Button>
            </Card>
          )}
        </div>
      </div>

      {/* Proposal Detail Modal */}
      {selectedProposal && (
        <ProposalDetailDialog
          proposal={selectedProposal}
          onClose={() => setSelectedProposal(null)}
        />
      )}
    </div>
  );
}

interface CreateProposalFormProps {
  targetUserId?: string;
  onCancel: () => void;
  onSuccess: () => void;
}

function CreateProposalForm({ targetUserId, onCancel, onSuccess }: CreateProposalFormProps) {
  const [formData, setFormData] = useState({
    targetUserId: targetUserId || '',
    proposedTime: '',
    locationName: '',
    lat: 0,
    lng: 0,
    activityType: 'walk',
    message: '',
  });
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: (data: any) => apiClient.post('/meetup-proposals', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
      toast.success('Meetup proposal sent!');
      onSuccess();
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to send proposal');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate(formData);
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-bold mb-4">Propose a Meetup</h3>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Label>When?</Label>
          <Input
            type="datetime-local"
            value={formData.proposedTime}
            onChange={(e) => setFormData({ ...formData, proposedTime: e.target.value })}
            required
          />
        </div>

        <div>
          <Label>Where?</Label>
          <Input
            placeholder="Park name or meeting spot"
            value={formData.locationName}
            onChange={(e) => setFormData({ ...formData, locationName: e.target.value })}
            required
          />
        </div>

        <div>
          <Label>Activity</Label>
          <select
            className="w-full p-2 border rounded-md"
            value={formData.activityType}
            onChange={(e) => setFormData({ ...formData, activityType: e.target.value })}
          >
            <option value="walk">Walk</option>
            <option value="park_visit">Park Visit</option>
            <option value="cafe">Café</option>
            <option value="hike">Hike</option>
            <option value="playdate">Playdate</option>
          </select>
        </div>

        <div>
          <Label>Message (optional)</Label>
          <Textarea
            placeholder="Add a friendly message..."
            value={formData.message}
            onChange={(e) => setFormData({ ...formData, message: e.target.value })}
            rows={3}
          />
        </div>

        <div className="flex gap-2">
          <Button type="button" variant="outline" onClick={onCancel} className="flex-1">
            Cancel
          </Button>
          <Button type="submit" disabled={createMutation.isPending} className="flex-1">
            {createMutation.isPending ? 'Sending...' : 'Send Proposal'}
          </Button>
        </div>
      </form>
    </Card>
  );
}

interface ProposalCardProps {
  proposal: MeetupProposal;
  isPast?: boolean;
  onSelect: () => void;
}

function ProposalCard({ proposal, isPast, onSelect }: ProposalCardProps) {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'confirmed':
        return <Badge className="bg-green-500">Confirmed</Badge>;
      case 'accepted':
        return <Badge className="bg-blue-500">Accepted</Badge>;
      case 'rejected':
        return <Badge variant="destructive">Declined</Badge>;
      case 'completed':
        return <Badge variant="secondary">Completed</Badge>;
      default:
        return <Badge variant="outline">Pending</Badge>;
    }
  };

  return (
    <Card className="p-4 hover:shadow-md transition-all cursor-pointer" onClick={onSelect}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Avatar className="w-10 h-10">
            {proposal.proposer.avatarUrl ? (
              <img src={proposal.proposer.avatarUrl} alt={proposal.proposer.handle} />
            ) : (
              <div className="bg-accent text-white flex items-center justify-center">
                {proposal.proposer.handle[0].toUpperCase()}
              </div>
            )}
          </Avatar>
          <div>
            <p className="font-semibold">@{proposal.proposer.handle}</p>
            <p className="text-xs text-muted-foreground">
              {proposal.activityType.charAt(0).toUpperCase() + proposal.activityType.slice(1)}
            </p>
          </div>
        </div>
        {getStatusBadge(proposal.status)}
      </div>

      <div className="space-y-2 text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Clock className="w-4 h-4" />
          {new Date(proposal.proposedTime).toLocaleString()}
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <MapPin className="w-4 h-4" />
          {proposal.locationName}
        </div>
        {proposal.message && (
          <div className="flex items-start gap-2 text-muted-foreground">
            <MessageSquare className="w-4 h-4 mt-0.5" />
            <p className="line-clamp-2">{proposal.message}</p>
          </div>
        )}
      </div>

      {isPast && proposal.feedback && (
        <div className="mt-3 pt-3 border-t">
          <div className="flex items-center gap-2">
            <Star className="w-4 h-4 text-accent fill-current" />
            <span className="text-sm font-semibold">
              {proposal.feedback.didMeetupHappen ? 'Meetup happened!' : 'Meetup didn\'t happen'}
            </span>
            {proposal.feedback.rating && (
              <span className="text-sm text-muted-foreground">• {proposal.feedback.rating}/5</span>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}

interface ProposalDetailDialogProps {
  proposal: MeetupProposal;
  onClose: () => void;
}

function ProposalDetailDialog({ proposal, onClose }: ProposalDetailDialogProps) {
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [didHappen, setDidHappen] = useState(true);
  const [rating, setRating] = useState([5]);
  const [feedbackTags, setFeedbackTags] = useState<string[]>([]);
  const [comments, setComments] = useState('');
  const queryClient = useQueryClient();

  const isPast = new Date(proposal.proposedTime) < new Date();
  const canLeaveFeedback = isPast && proposal.status === 'confirmed' && !proposal.feedback;

  const acceptMutation = useMutation({
    mutationFn: () => apiClient.post(`/meetup-proposals/${proposal.id}/accept`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
      toast.success('Proposal accepted!');
      onClose();
    },
  });

  const rejectMutation = useMutation({
    mutationFn: () => apiClient.post(`/meetup-proposals/${proposal.id}/reject`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
      toast.success('Proposal declined');
      onClose();
    },
  });

  const feedbackMutation = useMutation({
    mutationFn: (data: any) => apiClient.post(`/meetup-proposals/${proposal.id}/feedback`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
      toast.success('Thanks for your feedback!');
      onClose();
    },
  });

  const handleSubmitFeedback = () => {
    feedbackMutation.mutate({
      didMeetupHappen: didHappen,
      rating: didHappen ? rating[0] : undefined,
      feedbackTags: didHappen ? feedbackTags : undefined,
      comments,
    });
  };

  const toggleTag = (tag: string) => {
    setFeedbackTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  const suggestedTags = ['Great conversation', 'Pets got along', 'Would meet again', 'Ran late', 'Location was perfect'];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <Card className="max-w-lg w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold">Meetup Details</h3>
            <Button variant="ghost" size="sm" onClick={onClose}>✕</Button>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-muted-foreground" />
              <span>{new Date(proposal.proposedTime).toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="w-5 h-5 text-muted-foreground" />
              <span>{proposal.locationName}</span>
            </div>
            {proposal.message && (
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-sm">{proposal.message}</p>
              </div>
            )}
          </div>

          {proposal.status === 'pending' && (
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex-1"
                onClick={() => rejectMutation.mutate()}
                disabled={rejectMutation.isPending}
              >
                <ThumbsDown className="w-4 h-4 mr-2" />
                Decline
              </Button>
              <Button
                className="flex-1"
                onClick={() => acceptMutation.mutate()}
                disabled={acceptMutation.isPending}
              >
                <ThumbsUp className="w-4 h-4 mr-2" />
                Accept
              </Button>
            </div>
          )}

          {canLeaveFeedback && !showFeedbackForm && (
            <Button onClick={() => setShowFeedbackForm(true)} className="w-full">
              <Star className="w-4 h-4 mr-2" />
              Leave Feedback
            </Button>
          )}

          {showFeedbackForm && (
            <div className="space-y-4 border-t pt-4">
              <div>
                <Label className="mb-3 block">Did the meetup happen?</Label>
                <div className="flex gap-2">
                  <Button
                    variant={didHappen ? 'default' : 'outline'}
                    onClick={() => setDidHappen(true)}
                    className="flex-1"
                  >
                    Yes
                  </Button>
                  <Button
                    variant={!didHappen ? 'default' : 'outline'}
                    onClick={() => setDidHappen(false)}
                    className="flex-1"
                  >
                    No
                  </Button>
                </div>
              </div>

              {didHappen && (
                <>
                  <div>
                    <div className="flex justify-between mb-2">
                      <Label>Rating</Label>
                      <span className="text-sm font-semibold">{rating[0]}/5</span>
                    </div>
                    <Slider value={rating} onValueChange={setRating} min={1} max={5} step={1} />
                  </div>

                  <div>
                    <Label className="mb-2 block">Quick tags</Label>
                    <div className="flex flex-wrap gap-2">
                      {suggestedTags.map(tag => (
                        <Badge
                          key={tag}
                          variant={feedbackTags.includes(tag) ? 'default' : 'outline'}
                          className="cursor-pointer"
                          onClick={() => toggleTag(tag)}
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </>
              )}

              <div>
                <Label>Comments (optional)</Label>
                <Textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  rows={3}
                  placeholder="Any additional thoughts?"
                />
              </div>

              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setShowFeedbackForm(false)} className="flex-1">
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmitFeedback}
                  disabled={feedbackMutation.isPending}
                  className="flex-1"
                >
                  Submit Feedback
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
