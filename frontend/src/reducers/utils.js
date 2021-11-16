function requestStateHandler(thunk) {
  return (builder) => {
    builder
      .addCase(thunk.pending, (state, action) => {
        state.status = 'loading';
      })
      .addCase(thunk.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.data = action.payload;
      })
      .addCase(thunk.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      });
  };
}

export { requestStateHandler };
