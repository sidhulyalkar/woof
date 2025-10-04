import { create } from 'zustand';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

interface Modal {
  id: string;
  component: React.ComponentType<any>;
  props?: any;
}

interface UIState {
  // Camera state
  isCameraOpen: boolean;
  cameraMode: 'photo' | 'video' | 'story';

  // Modal state
  modals: Modal[];

  // Toast state
  toasts: Toast[];

  // Loading state
  isLoading: boolean;
  loadingMessage?: string;

  // Bottom sheet state
  isBottomSheetOpen: boolean;
  bottomSheetContent?: React.ReactNode;

  // Actions
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
  // Camera state
  isCameraOpen: false,
  cameraMode: 'photo',

  // Modal state
  modals: [],

  // Toast state
  toasts: [],

  // Loading state
  isLoading: false,
  loadingMessage: undefined,

  // Bottom sheet state
  isBottomSheetOpen: false,
  bottomSheetContent: undefined,

  // Camera actions
  openCamera: (mode = 'photo') =>
    set({ isCameraOpen: true, cameraMode: mode }),

  closeCamera: () =>
    set({ isCameraOpen: false }),

  setCameraMode: (mode) =>
    set({ cameraMode: mode }),

  // Modal actions
  openModal: (modal) =>
    set((state) => ({
      modals: [...state.modals, { ...modal, id: `modal-${modalIdCounter++}` }]
    })),

  closeModal: (id) =>
    set((state) => ({
      modals: state.modals.filter((modal) => modal.id !== id)
    })),

  closeAllModals: () =>
    set({ modals: [] }),

  // Toast actions
  showToast: (toast) =>
    set((state) => {
      const id = `toast-${toastIdCounter++}`;
      const newToast = { ...toast, id };

      // Auto-hide toast after duration
      if (toast.duration !== 0) {
        setTimeout(() => {
          set((state) => ({
            toasts: state.toasts.filter((t) => t.id !== id)
          }));
        }, toast.duration || 3000);
      }

      return { toasts: [...state.toasts, newToast] };
    }),

  hideToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((toast) => toast.id !== id)
    })),

  // Loading actions
  setLoading: (isLoading, message) =>
    set({ isLoading, loadingMessage: message }),

  // Bottom sheet actions
  openBottomSheet: (content) =>
    set({ isBottomSheetOpen: true, bottomSheetContent: content }),

  closeBottomSheet: () =>
    set({ isBottomSheetOpen: false, bottomSheetContent: undefined })
}));
