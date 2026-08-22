import { create } from 'zustand';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

interface Modal {
  id: string;
  component: React.ComponentType<Record<string, unknown>>;
  props?: Record<string, unknown>;
}

interface UIState {
  isCameraOpen: boolean;
  cameraMode: 'photo' | 'video' | 'story';
  modals: Modal[];
  toasts: Toast[];
  isLoading: boolean;
  loadingMessage?: string;
  isBottomSheetOpen: boolean;
  bottomSheetContent?: React.ReactNode;
  openCamera: (mode?: 'photo' | 'video' | 'story') => void;
  closeCamera: () => void;
  setCameraMode: (mode: 'photo' | 'video' | 'story') => void;
  openModal: (modal: Omit<Modal, 'id'>) => void;
  closeModal: (id: string) => void;
  closeAllModals: () => void;
  showToast: (toast: Omit<Toast, 'id'>) => void;
  hideToast: (id: string) => void;
  setLoading: (isLoading: boolean, message?: string) => void;
  openBottomSheet: (content: React.ReactNode) => void;
  closeBottomSheet: () => void;
}

let toastIdCounter = 0;
let modalIdCounter = 0;

export const useUIStore = create<UIState>((set) => ({
  isCameraOpen: false,
  cameraMode: 'photo',
  modals: [],
  toasts: [],
  isLoading: false,
  loadingMessage: undefined,
  isBottomSheetOpen: false,
  bottomSheetContent: undefined,
  openCamera: (mode = 'photo') => set({ isCameraOpen: true, cameraMode: mode }),
  closeCamera: () => set({ isCameraOpen: false }),
  setCameraMode: (mode) => set({ cameraMode: mode }),
  openModal: (modal) =>
    set((state) => ({
      modals: [...state.modals, { ...modal, id: `modal-${modalIdCounter++}` }],
    })),
  closeModal: (id) =>
    set((state) => ({
      modals: state.modals.filter((modal) => modal.id !== id),
    })),
  closeAllModals: () => set({ modals: [] }),
  showToast: (toast) =>
    set((state) => {
      const id = `toast-${toastIdCounter++}`;
      const newToast = { ...toast, id };
      if (toast.duration !== 0) {
        setTimeout(() => {
          set((current) => ({
            toasts: current.toasts.filter((item) => item.id !== id),
          }));
        }, toast.duration || 3000);
      }
      return { toasts: [...state.toasts, newToast] };
    }),
  hideToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((toast) => toast.id !== id),
    })),
  setLoading: (isLoading, message) => set({ isLoading, loadingMessage: message }),
  openBottomSheet: (content) => set({ isBottomSheetOpen: true, bottomSheetContent: content }),
  closeBottomSheet: () => set({ isBottomSheetOpen: false, bottomSheetContent: undefined }),
}));
