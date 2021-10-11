/* eslint-disable react/prop-types */
import React from 'react';
import MuiTextField from '@mui/material/TextField';

function TextField({ field: { name, value, onChange, onBlur }, form: { errors }, ...props }) {
  return (
    <MuiTextField
      name={name}
      variant="filled"
      onChange={onChange}
      onBlur={onBlur}
      value={value}
      margin="normal"
      helperText={errors[name]}
      error={Boolean(errors[name])}
      {...props}
    />
  );
}

export default TextField;
