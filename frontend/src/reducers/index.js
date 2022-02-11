import { combineReducers } from '@reduxjs/toolkit';

import appReducer from './app';
import recommendersReducer from './recommenders';
import settingsReducer from './settings';
import dialogsReducer from './dialogs';

import { repsysApi } from '../api';

export default combineReducers({
  app: appReducer,
  recommenders: recommendersReducer,
  settings: settingsReducer,
  dialogs: dialogsReducer,
  [repsysApi.reducerPath]: repsysApi.reducer,
});
