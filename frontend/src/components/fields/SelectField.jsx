/* eslint-disable react/prop-types */
import React from 'react';
import TextField from '@mui/material/TextField';

function SelectField({
  field: { name, value, onChange, onBlur },
  form: { errors },
  options,
  ...props
}) {
  return (
    <TextField
      select
      value={value}
      onChange={onChange}
      onBlur={onBlur}
      name={name}
      fullWidth
      helperText={errors[name]}
      error={Boolean(errors[name])}
      variant="outlined"
      margin="normal"
      SelectProps={{
        native: true,
      }}
      {...props}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </TextField>
  );
}

export default SelectField;
