/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';
import { customAlphabet } from 'nanoid';

const nanoid = customAlphabet('1234567890abcdef', 10);

export const slice = createSlice({
  name: 'recommenders',
  initialState: [],
  reducers: {
    addBar: {
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
    updateBarsOrder: (state, action) => {
      const { dragIndex, hoverIndex } = action.payload;
      [state[dragIndex], state[hoverIndex]] = [state[hoverIndex], state[dragIndex]];
    },
    updateBar: (state, action) => {
      const bar = state.find(({ id }) => id === action.payload.id);
      if (bar) {
        Object.assign(bar, action.payload);
      }
    },
    duplicateBar: {
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
    removeBar: (state, action) => {
      state.splice(action.payload, 1);
    },
  },
});

export const { addBar, removeBar, updateBarsOrder, duplicateBar, updateBar } = slice.actions;

export const recommendersSelector = (state) => state.recommenders;

export default slice.reducer;
