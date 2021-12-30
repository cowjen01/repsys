/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'root',
  initialState: {
    buildMode: false,
    selectedUser: null,
    favouriteUsers: [],
    sessionRecording: false,
    customInteractions: [],
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    toggleSessionRecording: (state, { payload }) => {
      state.sessionRecording = !state.sessionRecording;
      state.selectedUser = null;
      if (payload) {
        state.customInteractions = payload;
      }
    },
    setSelectedUser: (state, { payload }) => {
      state.selectedUser = payload;
    },
    toggleFavouriteUser: (state) => {
      if (state.favouriteUsers.includes(state.selectedUser)) {
        state.favouriteUsers = state.favouriteUsers.filter((user) => user !== state.selectedUser);
      } else {
        state.favouriteUsers.push(state.selectedUser);
      }
    },
    setCustomInteractions: (state, { payload }) => {
      state.customInteractions = payload;
    },
    addCustomInteraction: (state, { payload }) => {
      state.customInteractions.unshift(payload);
    },
  },
});

export const {
  toggleBuildMode,
  setSelectedUser,
  toggleSessionRecording,
  addCustomInteraction,
  setCustomInteractions,
  toggleFavouriteUser
} = slice.actions;

export const buildModeSelector = (state) => state.root.buildMode;

export const selectedUserSelector = (state) => state.root.selectedUser;

export const favouriteUsersSelector = (state) => state.root.favouriteUsers;

export const sessionRecordingSelector = (state) => state.root.sessionRecording;

export const customInteractionsSelector = (state) => state.root.customInteractions;

export default slice.reducer;
