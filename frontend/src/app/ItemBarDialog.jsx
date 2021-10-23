import React, { useMemo } from 'react';
import pt from 'prop-types';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import { Formik, Field } from 'formik';
import DialogTitle from '@mui/material/DialogTitle';

import TextField from './TextField';
import SelectField from './SelectField';

function ItemBarDialog({ open, onClose, initialValues, onSubmit, models }) {
  return (
    <Dialog open={open} fullWidth maxWidth="sm" onClose={onClose}>
      <DialogTitle>Recommender Configuration</DialogTitle>
      <Formik
        initialValues={initialValues}
        validate={(values) => {
          const errors = {};
          const requiredMessage = 'This field is required';
          if (!values.title) {
            errors.title = requiredMessage;
          }
          return errors;
        }}
        onSubmit={(values, { setSubmitting }) => {
          onSubmit(values);
          setSubmitting(false);
        }}
      >
        {({ submitForm, isSubmitting, values }) => {
          const model = useMemo(() => models.find((m) => m.key === values.model), [values.model]);
          return (
            <>
              <DialogContent>
                <Field name="title" label="Title" fullWidth component={TextField} />
                <Field
                  name="itemsPerPage"
                  fullWidth
                  label="Items per page"
                  component={SelectField}
                  options={[1, 2, 3, 4].map((i) => ({ label: i, value: i }))}
                />
                <Field
                  name="model"
                  fullWidth
                  label="Recommendation model"
                  component={SelectField}
                  options={[
                    // { label: 'Select model', value: 'null' },
                    ...models.map((m) => ({ label: m.key, value: m.key })),
                  ]}
                />
                {model &&
                  model.attributes &&
                  model.attributes.map((a) => (
                    <Field
                      key={a.key}
                      name={`modelAttributes.${values.model}.${a.key}`}
                      type={a.type || 'text'}
                      fullWidth
                      label={a.label}
                      component={TextField}
                    />
                  ))}
                {model && model.businessRules && model.businessRules.length && (
                  <Field
                    name={`businessRule.${values.model}`}
                    fullWidth
                    label="Business rule"
                    component={SelectField}
                    options={model.businessRules.map((b) => ({ label: b, value: b }))}
                  />
                )}
              </DialogContent>
              <DialogActions>
                <Button onClick={onClose} color="secondary">
                  Close
                </Button>
                <Button onClick={submitForm} color="secondary" disabled={isSubmitting} autoFocus>
                  Save
                </Button>
              </DialogActions>
            </>
          );
        }}
      </Formik>
    </Dialog>
  );
}

ItemBarDialog.defaultProps = {
  initialValues: {},
  models: [],
};

ItemBarDialog.propTypes = {
  open: pt.bool.isRequired,
  onClose: pt.func.isRequired,
  onSubmit: pt.func.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  initialValues: pt.any,
  // eslint-disable-next-line react/forbid-prop-types
  models: pt.any,
};

export default ItemBarDialog;
