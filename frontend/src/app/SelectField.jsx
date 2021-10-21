/* eslint-disable react/prop-types */
import React from 'react';
import TextField from '@mui/material/TextField';
// import MenuItem from '@mui/material/MenuItem';

function SelectField({
  field: { name, value, onChange, onBlur },
  form: { errors, touched },
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
      helperText={touched[name] && errors[name]}
      error={touched[name] && Boolean(errors[name])}
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
