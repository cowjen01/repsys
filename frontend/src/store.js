import { createStore, applyMiddleware, combineReducers } from '@reduxjs/toolkit';

import studioReducer from './app/studioSlice';
import layoutReducer from './app/layoutSlice';

const STORAGE_STATE_KEY = 'repsysApplicationState';

const rootReducer = combineReducers({
  studio: studioReducer,
  layout: layoutReducer,
});

export const saveStateMiddleware = (storeAPI) => (next) => (action) => {
  const result = next(action);
  localStorage.setItem(STORAGE_STATE_KEY, JSON.stringify(storeAPI.getState()));
  return result;
};

export const restoreState = () => {
  if (localStorage.getItem(STORAGE_STATE_KEY) !== null) {
    return JSON.parse(localStorage.getItem(STORAGE_STATE_KEY));
  }
  return {};
};

const middlewareEnhancer = applyMiddleware(saveStateMiddleware);

const store = createStore(rootReducer, restoreState(), middlewareEnhancer);

export default store;
