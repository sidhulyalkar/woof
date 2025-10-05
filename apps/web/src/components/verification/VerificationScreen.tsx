'use client';

import React, { useState, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Upload, FileText, CheckCircle, XCircle, Clock, Shield } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';

interface Verification {
  id: string;
  userId: string;
  petId?: string;
  documentType: string;
  fileUrl: string;
  status: 'pending' | 'approved' | 'rejected';
  notes?: string;
  reviewNotes?: string;
  uploadedAt: string;
  reviewedAt?: string;
  pet?: {
    id: string;
    name: string;
  };
}

export function VerificationScreen() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState('vaccination_record');
  const [petId, setPetId] = useState<string>('');
  const [notes, setNotes] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const { data: verifications, isLoading } = useQuery<Verification[]>({
    queryKey: ['verifications'],
    queryFn: () => apiClient.get<Verification[]>('/verification/me'),
  });

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      return apiClient.upload('/verification/upload', formData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['verifications'] });
      toast.success('Document uploaded successfully! We\'ll review it soon.');
      resetForm();
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to upload document');
    },
  });

  const resetForm = () => {
    setSelectedFile(null);
    setDocumentType('vaccination_record');
    setPetId('');
    setNotes('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
      if (!validTypes.includes(file.type)) {
        toast.error('Please upload a JPG, PNG, or PDF file');
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB');
        return;
      }

      setSelectedFile(file);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile) {
      toast.error('Please select a file to upload');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('documentType', documentType);
    if (petId) {
      formData.append('petId', petId);
    }
    if (notes) {
      formData.append('notes', notes);
    }

    uploadMutation.mutate(formData);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'approved':
        return (
          <Badge className="bg-green-500/10 text-green-700 border-green-500/20">
            <CheckCircle className="w-3 h-3 mr-1" />
            Approved
          </Badge>
        );
      case 'rejected':
        return (
          <Badge variant="destructive" className="bg-red-500/10 border-red-500/20">
            <XCircle className="w-3 h-3 mr-1" />
            Rejected
          </Badge>
        );
      default:
        return (
          <Badge variant="secondary">
            <Clock className="w-3 h-3 mr-1" />
            Pending Review
          </Badge>
        );
    }
  };

  const getDocumentTypeLabel = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Shield className="w-12 h-12 mx-auto mb-4 text-accent animate-pulse" />
          <p className="text-muted-foreground">Loading verifications...</p>
        </div>
      </div>
    );
  }

  const hasApprovedDoc = verifications?.some(v => v.status === 'approved');

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl font-bold">Safety Verification</h1>
            {hasApprovedDoc && <Shield className="w-6 h-6 text-green-500" />}
          </div>
          <p className="text-sm text-muted-foreground">
            Upload documents to verify your account and build trust
          </p>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Upload Form */}
          <Card className="p-6">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Upload New Document
            </h3>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Label htmlFor="documentType">Document Type *</Label>
                <Select value={documentType} onValueChange={setDocumentType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="vaccination_record">Vaccination Record</SelectItem>
                    <SelectItem value="vet_certificate">Vet Certificate</SelectItem>
                    <SelectItem value="license">Pet License</SelectItem>
                    <SelectItem value="identity">ID / Driver's License</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="file">File * (JPG, PNG, or PDF, max 10MB)</Label>
                <div className="mt-2">
                  <input
                    ref={fileInputRef}
                    id="file"
                    type="file"
                    accept="image/jpeg,image/png,image/jpg,application/pdf"
                    onChange={handleFileSelect}
                    className="block w-full text-sm text-muted-foreground
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-semibold
                      file:bg-accent file:text-white
                      hover:file:bg-accent/90
                      cursor-pointer"
                  />
                </div>
                {selectedFile && (
                  <p className="text-sm text-muted-foreground mt-2">
                    Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </p>
                )}
              </div>

              <div>
                <Label htmlFor="notes">Additional Notes (optional)</Label>
                <Textarea
                  id="notes"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Any additional information about this document..."
                  rows={3}
                />
              </div>

              <Button
                type="submit"
                disabled={uploadMutation.isPending || !selectedFile}
                className="w-full"
              >
                {uploadMutation.isPending ? 'Uploading...' : 'Upload Document'}
              </Button>
            </form>
          </Card>

          {/* Uploaded Documents */}
          <div>
            <h3 className="text-lg font-bold mb-4">Your Documents</h3>

            {!verifications || verifications.length === 0 ? (
              <Card className="p-8 text-center">
                <FileText className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                <h4 className="font-semibold mb-2">No documents uploaded yet</h4>
                <p className="text-sm text-muted-foreground">
                  Upload verification documents to build trust with the community
                </p>
              </Card>
            ) : (
              <div className="space-y-4">
                {verifications.map(verification => (
                  <Card key={verification.id} className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h4 className="font-semibold mb-1">
                          {getDocumentTypeLabel(verification.documentType)}
                        </h4>
                        <p className="text-sm text-muted-foreground">
                          Uploaded {new Date(verification.uploadedAt).toLocaleDateString()}
                        </p>
                        {verification.pet && (
                          <p className="text-sm text-muted-foreground">
                            For: {verification.pet.name}
                          </p>
                        )}
                      </div>
                      {getStatusBadge(verification.status)}
                    </div>

                    {verification.notes && (
                      <div className="mb-3">
                        <p className="text-sm text-muted-foreground">
                          <span className="font-medium">Your notes:</span> {verification.notes}
                        </p>
                      </div>
                    )}

                    {verification.reviewNotes && (
                      <div className="p-3 rounded-lg bg-muted/50">
                        <p className="text-sm">
                          <span className="font-medium">Review notes:</span> {verification.reviewNotes}
                        </p>
                      </div>
                    )}

                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-3"
                      onClick={() => window.open(verification.fileUrl, '_blank')}
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      View Document
                    </Button>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
