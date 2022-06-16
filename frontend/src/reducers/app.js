/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'app',
  initialState: {
    buildMode: false,
    selectedUser: null,
    favouriteUsers: [],
    interactiveMode: false,
    customInteractions: [],
    seenTutorials: [],
    initialized: false,
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    toggleInteractiveMode: (state, { payload }) => {
      state.interactiveMode = !state.interactiveMode;
      state.selectedUser = null;
      if (payload) {
        state.customInteractions = payload;
      }
    },
    setInitialized: (state) => {
      state.initialized = true;
    },
    setSelectedUser: (state, { payload }) => {
      state.selectedUser = payload;
    },
    addUserToFavourites: (state, { payload }) => {
      state.favouriteUsers.push(payload);
    },
    removeUserFromFavourites: (state, { payload }) => {
      state.favouriteUsers = state.favouriteUsers.filter((user) => user !== payload);
    },
    setCustomInteractions: (state, { payload }) => {
      state.customInteractions = payload;
    },
    addCustomInteraction: (state, { payload }) => {
      state.customInteractions.unshift(payload);
    },
    addSeenTutorial: (state, { payload }) => {
      state.seenTutorials.push(payload);
    },
  },
});

export const {
  toggleBuildMode,
  setSelectedUser,
  addUserToFavourites,
  removeUserFromFavourites,
  toggleInteractiveMode,
  addCustomInteraction,
  setCustomInteractions,
  addSeenTutorial,
  setInitialized,
} = slice.actions;

export const buildModeSelector = (state) => state.app.buildMode;

export const selectedUserSelector = (state) => state.app.selectedUser;

export const favouriteUsersSelector = (state) => state.app.favouriteUsers;

export const interactiveModeSelector = (state) => state.app.interactiveMode;

export const customInteractionsSelector = (state) => state.app.customInteractions;

export const seenTutorialsSelector = (state) => state.app.seenTutorials;

export const initializedSelector = (state) => state.app.initialized;

export default slice.reducer;
