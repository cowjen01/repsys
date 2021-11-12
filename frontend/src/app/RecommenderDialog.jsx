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
import CheckboxField from './CheckboxField';

function RecommenderDialog({ open, onClose, initialValues, onSubmit, models }) {
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
          if (!values.itemsLimit) {
            errors.itemsLimit = requiredMessage;
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
                  label="Items per page"
                  component={SelectField}
                  options={[1, 2, 3, 4].map((i) => ({ label: i, value: i }))}
                />
                <Field
                  name="itemsLimit"
                  label="Max number of items"
                  component={TextField}
                  type="number"
                />
                <Field
                  name="model"
                  label="Recommendation model"
                  component={SelectField}
                  options={[...models.map((m) => ({ label: m.key, value: m.key }))]}
                />
                {model &&
                  model.params &&
                  model.params.map((a) => {
                    const name = `modelParams.${values.model}.${a.key}`;
                    if (a.type === 'select') {
                      return (
                        <Field
                          key={a.key}
                          name={name}
                          label={a.label}
                          component={SelectField}
                          options={a.options.map((b) => ({ label: b, value: b }))}
                        />
                      );
                    }
                    if (a.type === 'bool') {
                      return (
                        <Field key={a.key} name={name} label={a.label} component={CheckboxField} />
                      );
                    }
                    return (
                      <Field
                        key={a.key}
                        name={name}
                        label={a.label}
                        component={TextField}
                        type={a.type}
                      />
                    );
                  })}
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

RecommenderDialog.defaultProps = {
  initialValues: {},
  models: [],
};

RecommenderDialog.propTypes = {
  open: pt.bool.isRequired,
  onClose: pt.func.isRequired,
  onSubmit: pt.func.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  initialValues: pt.any,
  // eslint-disable-next-line react/forbid-prop-types
  models: pt.any,
};

export default RecommenderDialog;
