import { useState, useEffect } from 'react';

export function getRequest(path, params = {}) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setIsLoading] = useState(true);

  const encodedParams = new URLSearchParams(params).toString();
  const fullPath = `/api${path}${encodedParams ? `?${encodedParams}` : ''}`;

  useEffect(() => {
    let isActive = true;

    setIsLoading(true);

    fetch(fullPath)
      .then((response) => response.json())
      .then((data) => {
        if (isActive) {
          setItems(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        setIsLoading(false);
        setError(err);
      });

    return () => {
      isActive = false;
    };
  }, [fullPath]);

  return { items, isLoading, error };
}

export function postRequest(path, body = {}) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setIsLoading] = useState(true);

  const encodedBody = JSON.stringify(body);
  const fullPath = `/api${path}`;

  useEffect(() => {
    let isActive = true;

    setIsLoading(true);

    fetch(fullPath, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: encodedBody,
    })
      .then((response) => response.json())
      .then((data) => {
        if (isActive) {
          setItems(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        setIsLoading(false);
        setError(err);
      });

    return () => {
      isActive = false;
    };
  }, [fullPath, encodedBody]);

  return { items, isLoading, error };
}
