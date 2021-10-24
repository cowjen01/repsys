import { createStore, applyMiddleware, combineReducers } from '@reduxjs/toolkit';

import studioReducer from './reducers/studio';
import layoutReducer from './reducers/layout';
import settingsReducer from './reducers/settings';

const STORAGE_STATE_KEY = 'repsysApplicationState';

const rootReducer = combineReducers({
  studio: studioReducer,
  layout: layoutReducer,
  settings: settingsReducer,
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
