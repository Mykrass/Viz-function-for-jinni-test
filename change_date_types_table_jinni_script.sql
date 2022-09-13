ALTER TABLE jinni ALTER COLUMN hire_reported TYPE TIMESTAMP WITH TIME ZONE USING hire_reported::timestamptz;
