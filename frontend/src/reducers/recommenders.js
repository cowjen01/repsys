/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';
import { customAlphabet } from 'nanoid';

const nanoid = customAlphabet('1234567890abcdef', 10);

export const slice = createSlice({
  name: 'recommenders',
  initialState: [],
  reducers: {
    addRecommender: {
      reducer(state, action) {
        state.push(action.payload);
      },
      prepare(values) {
        return {
          payload: {
            id: nanoid(),
            ...values,
          },
        };
      },
    },
    updateRecommendersOrder: (state, action) => {
      const { dragIndex, hoverIndex } = action.payload;
      [state[dragIndex], state[hoverIndex]] = [state[hoverIndex], state[dragIndex]];
    },
    updateRecommender: (state, action) => {
      const bar = state.find(({ id }) => id === action.payload.id);
      if (bar) {
        Object.assign(bar, action.payload);
      }
    },
    duplicateRecommender: {
      reducer(state, action) {
        const { index, id } = action.payload;
        const source = state[index];
        state.splice(index + 1, 0, {
          ...source,
          id,
          title: `${source.title} - copy`,
        });
      },
      prepare(index) {
        return {
          payload: {
            index,
            id: nanoid(),
          },
        };
      },
    },
    deleteRecommender: (state, action) => {
      state.splice(action.payload, 1);
    },
  },
});

export const {
  addRecommender,
  deleteRecommender,
  updateRecommendersOrder,
  duplicateRecommender,
  updateRecommender,
} = slice.actions;

export const recommendersSelector = (state) => state.recommenders;

export default slice.reducer;
